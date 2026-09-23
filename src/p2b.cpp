// p2b: 4x4x4 convolution (stride 4, no padding) over a 256^3 array
// stored in row-major order (A) and in morton order (B)

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "p2a.h"
#include "timer.h"

namespace {

constexpr size_t X = 256, Y = 256, Z = 256;  // input shape
constexpr size_t KS = 4;                     // kernel side length
constexpr size_t STRIDE = 4;
constexpr size_t KVOL = KS * KS * KS;  // 64 kernel entries
// output size = (M - K) / S + 1
constexpr size_t OX = (X - KS) / STRIDE + 1;
constexpr size_t OY = (Y - KS) / STRIDE + 1;
constexpr size_t OZ = (Z - KS) / STRIDE + 1;

// row-major indexing (z outermost, x innermost)
inline size_t rowMajorIndexA(size_t x, size_t y, size_t z) {
    assert(x < X && y < Y && z < Z);
    return (z * Y + y) * X + x;
}

inline size_t rowMajorIndexK(size_t x, size_t y, size_t z) {
    assert(x < KS && y < KS && z < KS);
    return (z * KS + y) * KS + x;
}

inline size_t rowMajorIndexConv(size_t x, size_t y, size_t z) {
    assert(x < OX && y < OY && z < OZ);
    return (z * OY + y) * OX + x;
}

// stops the optimizer from removing the convolution in release builds
inline void keepAlive(const void* p) { asm volatile("" : : "g"(p) : "memory"); }

// check morton3d against the handout examples
void testMorton() {
    assert(morton3d(0, 0, 0) == 0);
    assert(morton3d(1, 0, 0) == 0b001);
    assert(morton3d(0, 1, 0) == 0b010);
    assert(morton3d(0, 0, 1) == 0b100);
    assert(morton3d(0b0001, 0b0010, 0b0100) == 0b000100010001);
    // truncation past bit 63
    assert(morton3d(1ULL << 21, 0, 0) == 1ULL << 63);
    assert(morton3d(0, 1ULL << 21, 0) == 0);
    assert(morton3d(0, 0, 1ULL << 21) == 0);
    // 256^3 fits exactly in 2^24
    assert(morton3d(X - 1, Y - 1, Z - 1) == X * Y * Z - 1);
}

// row-major convolution: dot product of kernel with each 4x4x4 block
void convRowMajor(const std::vector<uint64_t>& a, const std::vector<uint64_t>& ka,
                  std::vector<uint64_t>& out) {
    assert(a.size() == X * Y * Z && ka.size() == KVOL);
    assert(out.size() == OX * OY * OZ);
    for (size_t oz = 0; oz < OZ; ++oz) {
        for (size_t oy = 0; oy < OY; ++oy) {
            for (size_t ox = 0; ox < OX; ++ox) {
                uint64_t dot = 0;
                for (size_t kz = 0; kz < KS; ++kz) {
                    for (size_t ky = 0; ky < KS; ++ky) {
                        for (size_t kx = 0; kx < KS; ++kx) {
                            dot += a[rowMajorIndexA(ox * STRIDE + kx, oy * STRIDE + ky,
                                                    oz * STRIDE + kz)] *
                                   ka[rowMajorIndexK(kx, ky, kz)];
                        }
                    }
                }
                out[rowMajorIndexConv(ox, oy, oz)] = dot;
            }
        }
    }
}

// morton convolution: each 4x4x4 block is 64 consecutive elements laid out
// like kb, so output m is just block m dotted with kb (no morton3d needed)
void convMorton(const std::vector<uint64_t>& b, const std::vector<uint64_t>& kb,
                std::vector<uint64_t>& out) {
    assert(b.size() == X * Y * Z && kb.size() == KVOL);
    assert(out.size() == OX * OY * OZ);
    for (size_t m = 0; m < out.size(); ++m) {
        const uint64_t* block = &b[m * KVOL];
        uint64_t dot = 0;
        for (size_t i = 0; i < KVOL; ++i) {
            dot += block[i] * kb[i];
        }
        out[m] = dot;
    }
}

}  // namespace

int main() {
    testMorton();

    // fill A in memory order
    std::vector<uint64_t> a(X * Y * Z);
    std::mt19937_64 gen(0);
    for (uint64_t& v : a) {
        v = gen();
    }

    // copy A into B in morton order
    std::vector<uint64_t> b(X * Y * Z);
    for (size_t z = 0; z < Z; ++z) {
        for (size_t y = 0; y < Y; ++y) {
            for (size_t x = 0; x < X; ++x) {
                b[morton3d(x, y, z)] = a[rowMajorIndexA(x, y, z)];
            }
        }
    }

    // kernel K(x,y,z) = x+y+z in both layouts
    std::vector<uint64_t> ka(KVOL), kb(KVOL);
    for (size_t z = 0; z < KS; ++z) {
        for (size_t y = 0; y < KS; ++y) {
            for (size_t x = 0; x < KS; ++x) {
                ka[rowMajorIndexK(x, y, z)] = x + y + z;
                kb[morton3d(x, y, z)] = x + y + z;
            }
        }
    }

    std::vector<uint64_t> outA(OX * OY * OZ), outB(OX * OY * OZ);

    Timer t;
    convRowMajor(a, ka, outA);
    keepAlive(outA.data());
    uint64_t timeA = t.click<Timer::Micros>();

    convMorton(b, kb, outB);
    keepAlive(outB.data());
    uint64_t timeB = t.click<Timer::Micros>();

    // compare results (outB is in morton order)
    bool same = true;
    for (size_t z = 0; z < OZ; ++z) {
        for (size_t y = 0; y < OY; ++y) {
            for (size_t x = 0; x < OX; ++x) {
                bool eq = outA[rowMajorIndexConv(x, y, z)] == outB[morton3d(x, y, z)];
                assert(eq);
                same = same && eq;
            }
        }
    }

    std::cout << timeA << '\n' << timeB << '\n';
    return same ? 0 : 1;
}
