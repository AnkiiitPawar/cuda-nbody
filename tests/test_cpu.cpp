#include <cmath>
#include <cstdio>
#include <vector>

#include "nb/nbody.hpp"
#include "test_util.hpp"

using namespace nb;
using testing::check;

int main() {
    {
        // m = 2 at distance 1 pulls with |a| = 2 / (1 + eps^2)^(3/2) towards it.
        std::vector<Body4> pos = {{-0.5f, 0, 0, 1.0f}, {0.5f, 0, 0, 2.0f}};
        std::vector<double> acc;
        const double eps = 1e-3;
        cpu::accelerations(pos, acc, eps);
        const double expect = 2.0 / std::pow(1.0 + eps * eps, 1.5);
        check(std::abs(acc[0] - expect) < 1e-12 && std::abs(acc[3] + 0.5 * expect) < 1e-12,
              "two-body accelerations match Newton's law");
    }
    {
        System s = plummer(1000, 1);
        std::vector<double> acc;
        cpu::accelerations(s.pos, acc, 0.01);
        double net[3] = {0, 0, 0}, scale = 0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            for (int k = 0; k < 3; ++k) net[k] += s.pos[i].w * acc[3 * i + k];
            scale += s.pos[i].w * std::sqrt(acc[3 * i] * acc[3 * i] + acc[3 * i + 1] * acc[3 * i + 1] +
                                            acc[3 * i + 2] * acc[3 * i + 2]);
        }
        check(std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]) < 1e-12 * scale,
              "internal forces cancel (momentum conservation)");
    }
    {
        System s = plummer(4000, 2);
        const double e = cpu::energy(s, 0.0);
        check(std::abs(e + 0.25) < 0.02, "Plummer sphere starts at E = -1/4 (Henon units)");
    }
    {
        System s = plummer(500, 3);
        const double eps = 0.05;
        std::vector<double> acc;
        cpu::accelerations(s.pos, acc, eps);
        const double e0 = cpu::energy(s, eps);
        for (int k = 0; k < 256; ++k) cpu::step(s, acc, 1.0f / 256.0f, eps);
        const double drift = std::abs(cpu::energy(s, eps) / e0 - 1.0);
        std::printf("  relative energy error after 1 time unit: %.2e\n", drift);
        check(drift < 1e-3, "leapfrog conserves energy over one time unit");
    }
    {
        System s = two_clusters(300, 4);
        double p[3] = {0, 0, 0};
        for (std::size_t i = 0; i < s.size(); ++i) {
            p[0] += s.pos[i].w * s.vel[i].x;
            p[1] += s.pos[i].w * s.vel[i].y;
            p[2] += s.pos[i].w * s.vel[i].z;
        }
        check(std::abs(p[0]) + std::abs(p[1]) + std::abs(p[2]) < 1e-5, "colliding clusters have zero net momentum");
    }
    std::printf("%d failure(s)\n", testing::failures);
    return testing::failures == 0 ? 0 : 1;
}
