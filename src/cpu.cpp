#include <cmath>
#include <cstddef>
#include <random>

#include "nb/nbody.hpp"

namespace nb {

System plummer(std::size_t n, std::uint64_t seed, float mass) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double pi = 3.14159265358979323846;
    auto isotropic = [&](double r, double out[3]) {
        const double z = 2.0 * u(rng) - 1.0, phi = 2.0 * pi * u(rng), s = std::sqrt(1.0 - z * z);
        out[0] = r * s * std::cos(phi);
        out[1] = r * s * std::sin(phi);
        out[2] = r * z;
    };
    const double length = 3.0 * pi / 16.0;           // Plummer scale -> Henon length unit
    const double speed = std::sqrt(mass / length);    // velocity scale for total mass `mass`
    System s;
    double com[3] = {0, 0, 0}, cov[3] = {0, 0, 0};
    for (std::size_t i = 0; i < n; ++i) {
        double r;
        do r = 1.0 / std::sqrt(std::pow(u(rng), -2.0 / 3.0) - 1.0);
        while (!(r < 20.0));
        double q, g;
        do {
            q = u(rng);
            g = 0.1 * u(rng);
        } while (g > q * q * std::pow(1.0 - q * q, 3.5));
        double x[3], v[3];
        isotropic(r * length, x);
        isotropic(q * std::sqrt(2.0) * std::pow(1.0 + r * r, -0.25) * speed, v);
        const float m = mass / static_cast<float>(n);
        s.pos.push_back({float(x[0]), float(x[1]), float(x[2]), m});
        s.vel.push_back({float(v[0]), float(v[1]), float(v[2]), 0.0f});
        for (int k = 0; k < 3; ++k) {
            com[k] += x[k] / n;
            cov[k] += v[k] / n;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        s.pos[i].x -= float(com[0]); s.pos[i].y -= float(com[1]); s.pos[i].z -= float(com[2]);
        s.vel[i].x -= float(cov[0]); s.vel[i].y -= float(cov[1]); s.vel[i].z -= float(cov[2]);
    }
    return s;
}

System two_clusters(std::size_t n_each, std::uint64_t seed) {
    System a = plummer(n_each, seed), b = plummer(n_each, seed + 1);
    System s;
    auto add = [&](const System& c, float dx, float dy, float dvx) {
        for (std::size_t i = 0; i < c.size(); ++i) {
            s.pos.push_back({c.pos[i].x + dx, c.pos[i].y + dy, c.pos[i].z, c.pos[i].w});
            s.vel.push_back({c.vel[i].x + dvx, c.vel[i].y, c.vel[i].z, 0.0f});
        }
    };
    add(a, -3.0f, -0.6f, +0.35f);
    add(b, +3.0f, +0.6f, -0.35f);
    return s;
}

namespace cpu {

void accelerations(const std::vector<Body4>& pos, std::vector<double>& acc, double eps) {
    const auto n = static_cast<std::ptrdiff_t>(pos.size());
    acc.assign(3 * pos.size(), 0.0);
    const double eps2 = eps * eps;
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        double ax = 0, ay = 0, az = 0;
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const double dx = double(pos[j].x) - pos[i].x, dy = double(pos[j].y) - pos[i].y,
                         dz = double(pos[j].z) - pos[i].z;
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double s = pos[j].w / (r2 * std::sqrt(r2));  // the j == i term is zero (dx = 0)
            ax += dx * s;
            ay += dy * s;
            az += dz * s;
        }
        acc[3 * i] = ax;
        acc[3 * i + 1] = ay;
        acc[3 * i + 2] = az;
    }
}

void step(System& s, std::vector<double>& acc, float dt, double eps) {
    const std::size_t n = s.size();
    for (std::size_t i = 0; i < n; ++i) {
        s.vel[i].x += float(0.5 * dt * acc[3 * i]);
        s.vel[i].y += float(0.5 * dt * acc[3 * i + 1]);
        s.vel[i].z += float(0.5 * dt * acc[3 * i + 2]);
        s.pos[i].x += dt * s.vel[i].x;
        s.pos[i].y += dt * s.vel[i].y;
        s.pos[i].z += dt * s.vel[i].z;
    }
    accelerations(s.pos, acc, eps);
    for (std::size_t i = 0; i < n; ++i) {
        s.vel[i].x += float(0.5 * dt * acc[3 * i]);
        s.vel[i].y += float(0.5 * dt * acc[3 * i + 1]);
        s.vel[i].z += float(0.5 * dt * acc[3 * i + 2]);
    }
}

double energy(const System& s, double eps) {
    const auto n = static_cast<std::ptrdiff_t>(s.size());
    double kinetic = 0, potential = 0;
    for (std::ptrdiff_t i = 0; i < n; ++i)
        kinetic += 0.5 * s.pos[i].w * (double(s.vel[i].x) * s.vel[i].x + double(s.vel[i].y) * s.vel[i].y +
                                       double(s.vel[i].z) * s.vel[i].z);
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 64)
    for (std::ptrdiff_t i = 0; i < n; ++i)
        for (std::ptrdiff_t j = i + 1; j < n; ++j) {
            const double dx = double(s.pos[j].x) - s.pos[i].x, dy = double(s.pos[j].y) - s.pos[i].y,
                         dz = double(s.pos[j].z) - s.pos[i].z;
            potential -= double(s.pos[i].w) * s.pos[j].w / std::sqrt(dx * dx + dy * dy + dz * dz + eps * eps);
        }
    return kinetic + potential;
}

}  // namespace cpu
}  // namespace nb
