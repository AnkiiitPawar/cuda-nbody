// Two star clusters colliding. Uses the GPU if one is available, otherwise the CPU reference.
// usage: collide [n_each] [output_csv]   -> CSV of (snapshot, time, x, y) for plotting

#include <cstdio>
#include <string>
#include <vector>

#include "nb/nbody.hpp"

using namespace nb;

int main(int argc, char** argv) {
    const int n_each = argc > 1 ? std::stoi(argv[1]) : 1500;
    const std::string out = argc > 2 ? argv[2] : "collision.csv";
    const float dt = 0.01f, eps = 0.05f;
    const int steps_per_snapshot = 300, snapshots = 4;  // t = 0, 3, 6, 9

    System s = two_clusters(n_each, 42);
    const bool gpu_ok = gpu::available();
    std::FILE* f = std::fopen(out.c_str(), "w");
    if (!f) return 1;
    std::fprintf(f, "snapshot,t,x,y\n");
    std::vector<double> acc;
    if (!gpu_ok) cpu::accelerations(s.pos, acc, eps);
    const double e0 = cpu::energy(s, eps);
    for (int k = 0; k < snapshots; ++k) {
        for (const Body4& b : s.pos) std::fprintf(f, "%d,%.2f,%.4f,%.4f\n", k, k * steps_per_snapshot * dt, b.x, b.y);
        if (k + 1 == snapshots) break;
        if (gpu_ok)
            gpu::integrate(s, dt, steps_per_snapshot, eps);
        else
            for (int i = 0; i < steps_per_snapshot; ++i) cpu::step(s, acc, dt, eps);
    }
    std::fclose(f);
    std::printf("%d bodies on the %s, relative energy error %.2e, wrote %s\n", 2 * n_each, gpu_ok ? "GPU" : "CPU",
                cpu::energy(s, eps) / e0 - 1.0, out.c_str());
    return 0;
}
