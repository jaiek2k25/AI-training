#include "pentanary.h"
#include <cassert>
#include <chrono>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

void loader_tests() {
    std::vector<uint8_t> bytes;
    auto put = [&](uint64_t v, int n) {
        for (int i = 0; i < n; ++i) bytes.push_back(uint8_t(v >> (8 * i)));
    };
    for (char c : std::string("PENTA5\0\0", 8)) bytes.push_back(uint8_t(c));
    put(1, 4); put(1, 4); put(128, 8);  // version, count, data start
    put(1, 2); put(1, 1); put(2, 1); put(0x3c00, 2); put(0, 2);
    put(0, 8); put(7, 8); put(5, 8); put(4, 8); put('w', 1);
    bytes.resize(128);
    for (int i = 0; i < 21; i += 3) {
        std::array<int8_t, 3> q{};
        for (int j = 0; j < 3; ++j) q[j] = i + j < 20 ? int8_t((i + j) % 5 - 2) : 0;
        bytes.push_back(pack_pentanary_byte(q));
    }
    char path[] = "/tmp/penta-test-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    assert(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()));
    close(fd);
    {
        Model model(path);
        const auto& t = model.tensor("w");
        float input[4] = {-1, -1, -1, -1}, output[5], row[4];
        model.matvec(t, input, 4, output, 5);
        for (int i = 0; i < 5; ++i) {
            model.row(t, i, row, 4);
            float expected = 0;
            for (int j = 0; j < 4; ++j) {
                assert(row[j] == float((4 * i + j) % 5 - 2));
                expected -= row[j];
            }
            assert(output[i] == expected);
        }
    }
    assert(truncate(path, 134) == 0);
    bool threw = false;
    try { Model invalid(path); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    unlink(path);
}

int main() {
    loader_tests();
    for (int b = 0; b < 125; ++b) {
        auto q = unpack_pentanary_byte(uint8_t(b));
        assert(pack_pentanary_byte(q) == b);
    }
    for (int b = 125; b < 256; ++b) {
        bool threw = false;
        try { (void)unpack_pentanary_byte(uint8_t(b)); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw);
    }
    for (int32_t a : {INT_MIN, -32767, -1, 0, 1, 32767, INT_MAX})
        for (int8_t w = -2; w <= 2; ++w)
            assert(pentanary_accumulate(123, a, w) == 123 + int64_t(a) * w);
    assert(half_to_float(0x3c00) == 1.0f);
    assert(half_to_float(0xc000) == -2.0f);
    assert(half_to_float(1) > 0);
    auto start = std::chrono::steady_clock::now();
    int64_t sum = 0;
    constexpr int count = 10000000;
    for (int i = 0; i < count; ++i) sum = pentanary_accumulate(sum, i % 65535 - 32767, int8_t(i % 5 - 2));
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "PASS: all 125 triplets, 131 invalid bytes, signed extremes, FP16 conversion\n"
              << count / seconds / 1e6 << " million synthetic state operations/s; checksum=" << sum << '\n';
}
