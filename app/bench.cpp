// Force-evaluation throughput: CPU (double, OpenMP) vs GPU naive vs GPU tiled.
// GFLOP/s uses the conventional 20 flops per pairwise interaction.

#include <chrono>
#include <cstdio>
#include <vector>

#include "nb/nbody.hpp"

using namespace nb;

int main() {
    const bool gpu_ok = gpu::available();
    const float eps = 0.01f;
    std::printf("%8s %22s %22s %22s\n", "N", "CPU double [ms|GF/s]", "GPU naive [ms|GF/s]", "GPU tiled [ms|GF/s]");
    for (int n : {1024, 4096, 16384, 65536}) {
        const System s = plummer(n, 1);
        const double interactions = double(n) * n;
        auto gflops = [&](double ms) { return 20.0 * interactions / (ms * 1e6); };
        char cpu_cell[64] = "-", naive_cell[64] = "-", tiled_cell[64] = "-";
        if (n <= 16384) {
            std::vector<double> acc;
            const auto t0 = std::chrono::steady_clock::now();
            cpu::accelerations(s.pos, acc, eps);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::snprintf(cpu_cell, sizeof cpu_cell, "%9.2f | %6.1f", ms, gflops(ms));
        }
        if (gpu_ok) {
            const double naive = gpu::time_accelerations(s.pos, eps, gpu::Kernel::Naive, 10);
            const double tiled = gpu::time_accelerations(s.pos, eps, gpu::Kernel::Tiled, 10);
            std::snprintf(naive_cell, sizeof naive_cell, "%9.3f | %6.0f", naive, gflops(naive));
            std::snprintf(tiled_cell, sizeof tiled_cell, "%9.3f | %6.0f", tiled, gflops(tiled));
        }
        std::printf("%8d %22s %22s %22s\n", n, cpu_cell, naive_cell, tiled_cell);
    }
    if (!gpu_ok) std::printf("\nno CUDA device found: GPU columns skipped\n");
    return 0;
}
