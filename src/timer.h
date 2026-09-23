#ifndef TIMER_H
#define TIMER_H
#include <chrono>   // steady_clock, time_point, durations
#include <cstdint>  // uint64_t

class Timer {
public:
    using Nanos = std::chrono::nanoseconds;
    using Micros = std::chrono::microseconds;
    using Millis = std::chrono::milliseconds;
    using Seconds = std::chrono::seconds;
    using Minutes = std::chrono::minutes;
    using Hours = std::chrono::hours;
    Timer() : start(std::chrono::steady_clock::now()){}
    void restart(){start = std::chrono::steady_clock::now();}

    template <typename T>
    uint64_t glance() const{
        return static_cast<uint64_t>(
            std::chrono::duration_cast<T>(std::chrono::steady_clock::now() - start).count());

    }

    template <typename T>
    uint64_t click() {
        auto now = std::chrono::steady_clock::now();
        auto elasped = std::chrono::duration_cast<T>(now - start).count();
        start = now;
        return static_cast<uint64_t>(elasped);
    }
private:
    std::chrono::steady_clock::time_point start;
};

#endif  // TIMER_H
