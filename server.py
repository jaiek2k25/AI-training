#!/usr/bin/env python3
"""Minimal OpenAI chat-completions subset; one active generation, local use."""
import argparse
import json
import time
import uuid
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from runtime import Runtime, rss_bytes
from memory_limits import cgroup_memory, require_hard_limit, STRICT_ADMISSION_BYTES


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'user', 'assistant']
    content: str = Field(max_length=65536)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'pentanary'
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=32, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    stream: bool = False
    stream_options: StreamOptions | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63-1)
    stop: str | list[str] | None = None
    n: Literal[1] = 1


class BodyLimit:
    """Limit actual bytes, including chunked requests, before JSON parsing."""
    def __init__(self, app, limit=1 << 20):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        parts, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            part = message.get('body', b'')
            size += len(part)
            if size > self.limit:
                response = JSONResponse({'error': {'message': 'request body exceeds 1 MiB', 'type': 'invalid_request_error'}}, status_code=413)
                return await response(scope, receive, send)
            parts.append(part)
            if not message.get('more_body', False):
                break
        body = b''.join(parts)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(runtime, model_name='pentanary'):
    app = FastAPI(title='Pentanary CPU inference', version='0.1.0')
    app.add_middleware(BodyLimit)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({'error': {'message': str(exc.detail), 'type': 'invalid_request_error', 'code': exc.status_code}}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({'error': {'message': str(exc), 'type': 'invalid_request_error', 'code': 422}}, status_code=422)

    @app.get('/', include_in_schema=False)
    def index():
        return RedirectResponse('/docs')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': model_name, 'rss_bytes': rss_bytes(),
                'kv_cache_bytes': runtime.kv_bytes, 'max_context': runtime.max_context,
                'memory_budget_bytes': runtime.memory_budget,
                'admission_estimate_bytes': runtime.admission_estimate,
                'strict_450mb_profile': getattr(runtime, 'strict_450mb', False),
                'cgroup_memory': cgroup_memory()}

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': model_name, 'object': 'model', 'created': 0, 'owned_by': 'local'}]}

    @app.post('/v1/chat/completions')
    def chat(req: ChatRequest):
        if req.model != model_name:
            raise HTTPException(404, 'unknown model')
        stop = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
        if len(stop) > 4 or any(not s or len(s) > 256 for s in stop):
            raise HTTPException(400, 'stop must contain 1–4 nonempty strings, at most 256 characters each')
        # Reject instead of accumulating queued requests and their contexts.
        if not runtime.lock.acquire(blocking=False):
            raise HTTPException(429, 'runtime busy; only one generation is supported')
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                runtime.lock.release()

        try:
            prompt = runtime.encode_messages([m.model_dump() for m in req.messages])
            if len(prompt) + req.max_tokens > runtime.max_context:
                raise ValueError('prompt + max_tokens exceeds max_context')
        except Exception as exc:
            release()
            raise HTTPException(400, str(exc)) from exc
        response_id, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())

        def base(object_type):
            return {'id': response_id, 'object': object_type, 'created': created, 'model': model_name}

        def usage(count):
            return {'prompt_tokens': len(prompt), 'completion_tokens': count, 'total_tokens': len(prompt) + count}

        def generate():
            return runtime.generate(prompt, req.max_tokens, req.temperature, req.top_p, req.seed, stop)

        if not req.stream:
            try:
                text, count, reason = '', 0, 'length'
                for delta, count, reason in generate():
                    text += delta
                return {**base('chat.completion'), 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': reason}], 'usage': usage(count)}
            except (ValueError, MemoryError, FloatingPointError) as exc:
                raise HTTPException(500, str(exc)) from exc
            finally:
                release()

        def stream():
            def frame(delta, reason=None):
                data = {**base('chat.completion.chunk'), 'choices': [{'index': 0, 'delta': delta, 'finish_reason': reason}]}
                return 'data: ' + json.dumps(data, ensure_ascii=False) + '\n\n'
            try:
                yield frame({'role': 'assistant', 'content': ''})
                count = 0
                for delta, count, reason in generate():
                    if delta:
                        yield frame({'content': delta})
                    if reason:
                        yield frame({}, reason)
                if req.stream_options and req.stream_options.include_usage:
                    yield 'data: ' + json.dumps({**base('chat.completion.chunk'), 'choices': [], 'usage': usage(count)}) + '\n\n'
                yield 'data: [DONE]\n\n'
            except (ValueError, MemoryError, FloatingPointError) as exc:
                yield 'data: ' + json.dumps({'error': {'message': str(exc), 'type': 'server_error'}}) + '\n\n'
                yield 'data: [DONE]\n\n'
            finally:
                release()
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
                                 background=BackgroundTask(release))
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='model_pentanary.bin')
    p.add_argument('--config', required=True, help='original HF config.json')
    p.add_argument('--tokenizer', required=True, help='original HF tokenizer.json')
    p.add_argument('--model-name', default='pentanary')
    p.add_argument('--max-context', type=int, default=512)
    p.add_argument('--memory-budget-mb', type=float, default=None,
                   help='soft process admission budget: default 480 MB, or 420 MB in strict profile')
    p.add_argument('--strict-450mb', action='store_true',
                   help='require cgroup-v2 hard cap <=450,000,000 bytes, swap disabled; no soft fallback')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    import math
    budget_mb = args.memory_budget_mb if args.memory_budget_mb is not None else (420 if args.strict_450mb else 480)
    if not math.isfinite(budget_mb) or not 0 < budget_mb < 500:
        p.error('memory budget must be finite, positive, and below 500 decimal MB')
    budget = int(budget_mb * 1_000_000)
    if args.strict_450mb:
        if budget > STRICT_ADMISSION_BYTES:
            p.error('strict profile reserves headroom: soft admission budget must be <=420 MB')
        try:
            require_hard_limit()
        except RuntimeError as exc:
            p.error(str(exc))
    engine = Runtime(args.model, args.config, args.tokenizer, args.max_context, budget)
    engine.strict_450mb = args.strict_450mb
    import uvicorn
    uvicorn.run(create_app(engine, args.model_name), host=args.host, port=args.port,
                workers=1, limit_concurrency=8, timeout_keep_alive=5)


if __name__ == '__main__':
    main()
