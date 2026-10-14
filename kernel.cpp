#include "pentanary.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

int64_t pentanary_accumulate(int64_t sum, int32_t a, int8_t w) {
    switch (w) {
        case 0: return sum;
        case 1: return sum + a;
        case -1: return sum - a;
        case 2:
        case -2: {
            // Left shifting negative signed integers is UB in C++17. Shift an
            // unsigned magnitude, restore sign in int64, then add/subtract.
            uint64_t magnitude = a < 0 ? uint64_t(-int64_t(a)) : uint64_t(a);
            int64_t doubled = int64_t(magnitude << 1);
            if (a < 0) doubled = -doubled;
            return w == 2 ? sum + doubled : sum - doubled;
        }
        default: throw std::runtime_error("invalid pentanary state");
    }
}
void Model::matvec(const Tensor& t, const float* x, uint64_t n, float* out, uint64_t m) const {
    if (!x || !out || t.kind != 1 || t.shape.size() != 2 || t.shape[0] != m || t.shape[1] != n
        || n > (1 << 24)) throw std::runtime_error("invalid matvec dimensions/type");
    float peak = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) throw std::runtime_error("nonfinite activation");
        peak = std::max(peak, std::abs(x[i]));
    }
    // Dynamic symmetric int16-range activation quantization; accumulated in
    // int64. Input/output scaling is floating point, inner weight MAC is not.
    // Double scale avoids underflow for float32 subnormal inputs.
    double scale = peak == 0 ? 1.0 : double(peak) / 32767.0;
    std::vector<int32_t> activations(n);
    for (uint64_t i = 0; i < n; ++i)
        activations[i] = int32_t(std::round(double(x[i]) / scale));
    std::array<uint8_t, 65536> buffer{};
    uint64_t col = 0, row = 0;
    int64_t sum = 0;
    for (uint64_t pos = 0; pos < t.bytes;) {
        uint64_t size = std::min<uint64_t>(buffer.size(), t.bytes - pos);
        read(t.offset + pos, buffer.data(), size);
        for (uint64_t b = 0; b < size; ++b) {
            auto q = unpack_pentanary_byte(buffer[b]);
            // Packed zero triplet (byte 62): no activation loads or adds.
            for (unsigned k = 0; k < 3 && row < m; ++k) {
                if (q[k]) sum = pentanary_accumulate(sum, activations[col], q[k]);
                if (++col == n) {
                    out[row++] = float(double(sum) * scale * t.alpha);
                    col = 0; sum = 0;
                }
            }
        }
        pos += size;
    }
}
