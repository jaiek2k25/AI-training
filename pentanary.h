#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

std::array<int8_t, 3> unpack_pentanary_byte(uint8_t byte);
uint8_t pack_pentanary_byte(const std::array<int8_t, 3>& weights);
float half_to_float(uint16_t bits);
struct Tensor {
    std::vector<uint64_t> shape;
    uint8_t kind;
    uint16_t flags;
    float alpha;
    uint64_t offset, bytes, count;
};
class Model {
public:
    explicit Model(const char* path);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    const Tensor& tensor(const std::string& name) const;
    void read(uint64_t offset, void* dest, uint64_t bytes) const;
    void row(const Tensor& t, uint64_t index, float* out, uint64_t size) const;
    void matvec(const Tensor& t, const float* x, uint64_t n, float* out, uint64_t m) const;
private:
    int fd_ = -1;
    uint64_t data_start_ = 0, file_size_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
};
// Zero states skip accumulator arithmetic (decoding/branching are NOT zero-cost).
int64_t pentanary_accumulate(int64_t accumulator, int32_t activation, int8_t weight);
extern "C" {
void* penta_open(const char* path);
void penta_close(void* model);
const char* penta_error();
int penta_matvec(void* model, const char* name, const float* x, uint64_t n, float* y, uint64_t m);
int penta_row(void* model, const char* name, uint64_t index, float* y, uint64_t n);
}
