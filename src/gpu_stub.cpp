// CPU-only build: GPU entry points exist but report that no device is available.

#include <stdexcept>

#include "nb/nbody.hpp"

namespace nb::gpu {

bool available() { return false; }

std::vector<Body4> accelerations(const std::vector<Body4>&, float, Kernel) {
    throw std::runtime_error("built without CUDA");
}

double time_accelerations(const std::vector<Body4>&, float, Kernel, int) {
    throw std::runtime_error("built without CUDA");
}

void integrate(System&, float, int, float) { throw std::runtime_error("built without CUDA"); }

}  // namespace nb::gpu
