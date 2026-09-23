// p1: row-major vs column-major traversal
// same random numbers summed in the same order, different memory access pattern

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "timer.h"

namespace {

constexpr size_t N = 4000;  // side length; each array holds N*N elements

// row-major index into an N x N array
inline size_t idx(size_t row, size_t col) {
    assert(row < N && col < N);
    return row * N + col;
}

}  // namespace

int main() {
    // contiguous, freed automatically
    std::vector<uint64_t> a(N * N), b(N * N);

    // same random values, different fill orders
    std::mt19937_64 gen(0);
    for (size_t r = 0; r < N; ++r) {  // a: row-major
        for (size_t c = 0; c < N; ++c) {
            a[idx(r, c)] = gen();
        }
    }
    gen.seed(0);
    for (size_t c = 0; c < N; ++c) {  // b: column-major
        for (size_t r = 0; r < N; ++r) {
            b[idx(r, c)] = gen();
        }
    }

    // row-major sum (sequential access)
    Timer t;
    uint64_t sumA = 0;
    for (size_t r = 0; r < N; ++r) {
        for (size_t c = 0; c < N; ++c) {
            sumA += a[idx(r, c)];
        }
    }
    uint64_t timeA = t.click<Timer::Micros>();

    // column-major sum (stride of N)
    uint64_t sumB = 0;
    for (size_t c = 0; c < N; ++c) {
        for (size_t r = 0; r < N; ++r) {
            sumB += b[idx(r, c)];
        }
    }
    uint64_t timeB = t.click<Timer::Micros>();

    assert(sumA == sumB);  // sums must match

    std::cout << timeA << ' ' << sumA << '\n';
    std::cout << timeB << ' ' << sumB << '\n';
    return 0;
}
