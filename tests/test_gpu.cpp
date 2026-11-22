// GPU tests. Without a CUDA device they exit with 77, which CTest reports as "skipped".

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "nb/nbody.hpp"
#include "test_util.hpp"

using namespace nb;
using testing::check;

int main() {
    if (!gpu::available()) {
        std::printf("no CUDA device: skipping GPU tests\n");
        return 77;
    }
    const float eps = 0.01f;
    System s = plummer(5000, 7);  // deliberately not a multiple of the tile size
    std::vector<double> ref;
    cpu::accelerations(s.pos, ref, eps);
    const auto naive = gpu::accelerations(s.pos, eps, gpu::Kernel::Naive);
    const auto tiled = gpu::accelerations(s.pos, eps, gpu::Kernel::Tiled);

    double worst_naive = 0, worst_tiled_vs_naive = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double ax = ref[3 * i], ay = ref[3 * i + 1], az = ref[3 * i + 2];
        const double norm = std::sqrt(ax * ax + ay * ay + az * az);
        const double dn = std::sqrt(std::pow(naive[i].x - ax, 2) + std::pow(naive[i].y - ay, 2) + std::pow(naive[i].z - az, 2));
        const double dt = std::sqrt(std::pow(tiled[i].x - naive[i].x, 2) + std::pow(tiled[i].y - naive[i].y, 2) +
                                    std::pow(tiled[i].z - naive[i].z, 2));
        worst_naive = std::max(worst_naive, dn / norm);
        worst_tiled_vs_naive = std::max(worst_tiled_vs_naive, dt / norm);
    }
    std::printf("  max relative error: naive vs CPU double %.2e, tiled vs naive %.2e\n", worst_naive, worst_tiled_vs_naive);
    check(worst_naive < 1e-3, "naive kernel matches the double-precision CPU reference");
    check(worst_tiled_vs_naive < 1e-5, "tiled kernel matches the naive kernel");

    System g = plummer(2048, 8);
    const double e0 = cpu::energy(g, eps);
    gpu::integrate(g, 1.0f / 256.0f, 256, eps);
    const double drift = std::abs(cpu::energy(g, eps) / e0 - 1.0);
    std::printf("  GPU leapfrog relative energy error after 1 time unit: %.2e\n", drift);
    check(drift < 1e-3, "GPU integration conserves energy");

    std::printf("%d failure(s)\n", testing::failures);
    return testing::failures == 0 ? 0 : 1;
}
