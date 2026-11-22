#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nb {

// Position + mass in one 16-byte record, layout-compatible with CUDA's float4.
// Velocities use the same type with w unused.
struct alignas(16) Body4 {
    float x, y, z, w;
};

struct System {
    std::vector<Body4> pos;  // x, y, z, mass
    std::vector<Body4> vel;  // vx, vy, vz, unused
    std::size_t size() const { return pos.size(); }
};

// Units G = 1. Plummer sphere of total mass `mass`, sampled as in Aarseth, Henon & Wielen (1974),
// scaled to Henon units (E = -1/4 for unit mass), in its centre-of-mass frame.
System plummer(std::size_t n, std::uint64_t seed, float mass = 1.0f);

// Two Plummer spheres on a slightly off-centre collision course.
System two_clusters(std::size_t n_each, std::uint64_t seed);

namespace cpu {
// Double-precision reference: a_i = sum_j m_j r_ij / (|r_ij|^2 + eps^2)^(3/2). acc holds 3n values.
void accelerations(const std::vector<Body4>& pos, std::vector<double>& acc, double eps);
// One kick-drift-kick leapfrog step; acc must hold the accelerations at the current positions.
void step(System& s, std::vector<double>& acc, float dt, double eps);
double energy(const System& s, double eps);
}  // namespace cpu

namespace gpu {
enum class Kernel { Naive, Tiled };
bool available();
std::vector<Body4> accelerations(const std::vector<Body4>& pos, float eps, Kernel kernel);
// Milliseconds per force evaluation, averaged over `reps` after one warm-up run.
double time_accelerations(const std::vector<Body4>& pos, float eps, Kernel kernel, int reps);
// `steps` leapfrog steps entirely on the device (tiled kernel), then copy the state back.
void integrate(System& s, float dt, int steps, float eps);
}  // namespace gpu

}  // namespace nb
