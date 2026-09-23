#ifndef P2A_H
#define P2A_H

#include <cstdint>

// moves bit i of v to bit 3i, bits past 63 are dropped
inline uint64_t expand(uint64_t v) {
    uint64_t result = 0;
    for (unsigned i = 0; 3 * i < 64; ++i) {
        result |= ((v >> i) & 1ULL) << (3 * i);
    }
    return result;
}

// 3D morton code: interleave bits of x, y, z (truncated to 64 bits)
inline uint64_t morton3d(uint64_t x, uint64_t y, uint64_t z) {
    return expand(x) | (expand(y) << 1) | (expand(z) << 2);
}

#endif  // P2A_H
