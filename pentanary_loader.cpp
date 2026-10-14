#include "pentanary.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
uint64_t le(const uint8_t* p, unsigned n) {
    uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}
thread_local std::string error;
}

std::array<int8_t, 3> unpack_pentanary_byte(uint8_t byte) {
    if (byte >= 125) throw std::runtime_error("invalid base-5 byte (must be <125)");
    return {int8_t(int(byte % 5) - 2), int8_t(int(byte / 5 % 5) - 2), int8_t(int(byte / 25) - 2)};
}
uint8_t pack_pentanary_byte(const std::array<int8_t, 3>& w) {
    for (auto v : w) if (v < -2 || v > 2) throw std::runtime_error("invalid pentanary state");
    return uint8_t((w[0] + 2) + 5 * (w[1] + 2) + 25 * (w[2] + 2));
}
float half_to_float(uint16_t h) {
    unsigned e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else v = std::ldexp(float(1024 + m), int(e) - 25);
    return h & 0x8000 ? -v : v;
}
void Model::read(uint64_t offset, void* dest, uint64_t bytes) const {
    if (offset > file_size_ || bytes > file_size_ - offset) throw std::runtime_error("read outside model file");
    auto* p = static_cast<uint8_t*>(dest);
    while (bytes) {
        ssize_t n = pread(fd_, p, std::min<uint64_t>(bytes, 1 << 20), off_t(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("short/failed model read");
        p += n; offset += uint64_t(n); bytes -= uint64_t(n);
    }
}
Model::Model(const char* path) {
    fd_ = open(path, O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("cannot open model");
    try {
        struct stat st{};
        if (fstat(fd_, &st) || !S_ISREG(st.st_mode)) throw std::runtime_error("model must be a regular file");
        file_size_ = uint64_t(st.st_size);
        uint8_t header[24]; read(0, header, 24);
        if (std::memcmp(header, "PENTA5\0\0", 8) || le(header + 8, 4) != 1) throw std::runtime_error("invalid PENTA5 header");
        uint64_t count = le(header + 12, 4);
        data_start_ = le(header + 16, 8);
        if (!count || count > 100000 || data_start_ < 24 || data_start_ > file_size_ || data_start_ > (64 << 20))
            throw std::runtime_error("invalid directory bounds");
        uint64_t pos = 24, last_end = 0;
        for (uint64_t i = 0; i < count; ++i) {
            uint8_t rec[24]; read(pos, rec, 24); pos += 24;
            uint64_t name_len = le(rec, 2), rank = rec[3];
            Tensor t{}; t.kind = rec[2]; t.alpha = half_to_float(uint16_t(le(rec + 4, 2)));
            t.flags = uint16_t(le(rec + 6, 2)); t.offset = le(rec + 8, 8); t.bytes = le(rec + 16, 8);
            if (!name_len || name_len > 4096 || !rank || rank > 8 || (t.kind != 1 && t.kind != 2)
                || (t.flags & ~1) || !std::isfinite(t.alpha) || t.alpha <= 0
                || pos > data_start_ || rank * 8 + name_len > data_start_ - pos)
                throw std::runtime_error("invalid tensor descriptor");
            t.count = 1;
            for (uint64_t j = 0; j < rank; ++j) {
                uint8_t dim[8]; read(pos, dim, 8); pos += 8;
                uint64_t d = le(dim, 8);
                if (!d || d > (1ULL << 40) / t.count) throw std::runtime_error("invalid tensor dimensions");
                t.count *= d; t.shape.push_back(d);
            }
            std::string name(name_len, '\0'); read(pos, name.data(), name_len); pos += name_len;
            uint64_t expected = t.kind == 1 ? (t.count + 2) / 3 : t.count * 2;
            if (name.find('\0') != std::string::npos || t.bytes != expected || t.offset < last_end
                || t.offset > file_size_ - data_start_ || t.bytes > file_size_ - data_start_ - t.offset)
                throw std::runtime_error("invalid payload bounds");
            last_end = t.offset + t.bytes;
            t.offset += data_start_;
            if (!tensors_.emplace(name, std::move(t)).second) throw std::runtime_error("duplicate tensor name");
        }
        // pread buffers bound user-space resident weights; OS page cache is separate.
        posix_fadvise(fd_, off_t(data_start_), 0, POSIX_FADV_SEQUENTIAL);
    } catch (...) { close(fd_); fd_ = -1; throw; }
}
Model::~Model() { if (fd_ >= 0) close(fd_); }
const Tensor& Model::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}
void Model::row(const Tensor& t, uint64_t index, float* out, uint64_t size) const {
    if (!out || t.shape.size() > 2) throw std::runtime_error("row expects vector or matrix");
    uint64_t width = t.shape.back(), rows = t.count / width;
    if (index >= rows || size != width) throw std::runtime_error("invalid row dimensions");
    // Bounded blocks, even for arbitrarily wide rows.
    if (t.kind == 2) {
        std::array<uint8_t, 65536> buffer{};
        for (uint64_t j = 0; j < width;) {
            uint64_t n = std::min<uint64_t>(width - j, buffer.size() / 2);
            read(t.offset + 2 * (index * width + j), buffer.data(), n * 2);
            for (uint64_t k = 0; k < n; ++k) {
                float v = half_to_float(uint16_t(le(buffer.data() + 2 * k, 2)));
                if (!std::isfinite(v)) throw std::runtime_error("nonfinite FP16 payload");
                out[j + k] = v;
            }
            j += n;
        }
        return;
    }
    uint64_t begin = index * width, first = begin / 3, finish = (begin + width + 2) / 3;
    std::array<uint8_t, 65536> buffer{};
    for (uint64_t b = first; b < finish;) {
        uint64_t n = std::min<uint64_t>(finish - b, buffer.size());
        read(t.offset + b, buffer.data(), n);
        for (uint64_t j = 0; j < n; ++j) {
            auto q = unpack_pentanary_byte(buffer[j]);
            for (unsigned k = 0; k < 3; ++k) {
                uint64_t at = 3 * (b + j) + k;
                if (at >= begin && at < begin + width) out[at - begin] = t.alpha * q[k];
            }
        }
        b += n;
    }
}
extern "C" {
void* penta_open(const char* path) {
    try { if (!path) throw std::runtime_error("null model path"); error.clear(); return new Model(path); }
    catch (const std::exception& e) { error = e.what(); return nullptr; }
}
void penta_close(void* model) { delete static_cast<Model*>(model); }
const char* penta_error() { return error.c_str(); }
int penta_row(void* model, const char* name, uint64_t index, float* y, uint64_t n) {
    try {
        if (!model || !name) throw std::runtime_error("null model/name");
        auto* m = static_cast<Model*>(model); m->row(m->tensor(name), index, y, n); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
int penta_matvec(void* model, const char* name, const float* x, uint64_t n, float* y, uint64_t rows) {
    try {
        if (!model || !name) throw std::runtime_error("null model/name");
        auto* m = static_cast<Model*>(model); m->matvec(m->tensor(name), x, n, y, rows); return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}
}
