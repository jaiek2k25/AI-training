#!/usr/bin/env python3
"""Launch the API in a systemd user scope with a verified 450 MB memory cap.

Requires Linux cgroup v2, a running user systemd manager and delegated memory
control. Failure is fatal: no unbounded fallback. Use --dry-run to inspect.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from memory_limits import page_aligned_limit

THREAD_ENV = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2'}


def scope_command(command, user=True):
    return ['systemd-run', *(['--user'] if user else []), '--scope', '--quiet',
            f'--property=MemoryMax={page_aligned_limit()}', '--property=MemorySwapMax=0',
            '--property=OOMPolicy=kill', '--', *command]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', default=sys.executable, help='inference venv interpreter')
    p.add_argument('--dry-run', action='store_true', help='print command only; applies no limits')
    p.add_argument('server_args', nargs=argparse.REMAINDER, help='-- followed by server.py arguments')
    args = p.parse_args(argv)
    server_args = args.server_args[1:] if args.server_args[:1] == ['--'] else args.server_args
    # Do not resolve interpreter symlinks: that would bypass the venv.
    python = os.path.abspath(args.python)
    command = scope_command([python, str(Path(__file__).with_name('server.py').absolute()),
                             '--strict-450mb', *server_args])
    if args.dry_run:
        print(json.dumps({'command': command, 'environment_overrides': THREAD_ENV,
                          'note': 'Dry run only; no cgroup created or limit enforced.'}, indent=2))
        return
    if not shutil.which('systemd-run'):
        p.error('systemd-run not available; provision a cgroup-v2 limit before using strict mode')
    if not os.access(python, os.X_OK):
        p.error('inference Python interpreter is not executable')
    try:
        # Scope creation happens before the API process starts. This launcher
        # uses no shell and imports no numpy/tokenizer/framework packages.
        os.execvpe(command[0], command, {**os.environ, **THREAD_ENV})
    except OSError as exc:
        p.exit(1, f'Cannot launch memory-limited scope: {exc}\n')


if __name__ == '__main__':
    main()
