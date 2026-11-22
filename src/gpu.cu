// Direct-summation gravity on the GPU: a naive kernel that streams every body from global
// memory, and a tiled kernel that stages blocks of bodies through shared memory so each
// global load is reused by every thread in the block (after Nyland, Harris & Prins, 2007).

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "nb/nbody.hpp"

namespace nb::gpu {
namespace {

constexpr int kTile = 256;  // threads per block = bodies per shared-memory tile

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Acceleration on body i from body j, with Plummer softening.
// About 20 floating-point operations: the conventional count for GFLOP/s figures.
__device__ __forceinline__ float3 interact(float4 bi, float4 bj, float3 a, float eps2) {
    const float dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
    const float r2 = dx * dx + dy * dy + dz * dz + eps2;
    const float inv = rsqrtf(r2);
    const float s = bj.w * inv * inv * inv;
    a.x += dx * s;
    a.y += dy * s;
    a.z += dz * s;
    return a;
}

__global__ void accel_naive(const float4* __restrict__ pos, float4* __restrict__ acc, int n, float eps2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 bi = pos[i];
    float3 a = make_float3(0.f, 0.f, 0.f);
    for (int j = 0; j < n; ++j) a = interact(bi, pos[j], a, eps2);
    acc[i] = make_float4(a.x, a.y, a.z, 0.f);
}

__global__ void accel_tiled(const float4* __restrict__ pos, float4* __restrict__ acc, int n, float eps2) {
    __shared__ float4 tile[kTile];
    const int i = blockIdx.x * kTile + threadIdx.x;
    const float4 bi = i < n ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3 a = make_float3(0.f, 0.f, 0.f);
    for (int base = 0; base < n; base += kTile) {
        const int j = base + threadIdx.x;
        // each thread loads one body; padding bodies get zero mass and contribute nothing
        tile[threadIdx.x] = j < n ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();
#pragma unroll 16
        for (int k = 0; k < kTile; ++k) a = interact(bi, tile[k], a, eps2);
        __syncthreads();
    }
    if (i < n) acc[i] = make_float4(a.x, a.y, a.z, 0.f);
}

__global__ void kick(float4* vel, const float4* acc, int n, float half_dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vel[i].x += half_dt * acc[i].x;
    vel[i].y += half_dt * acc[i].y;
    vel[i].z += half_dt * acc[i].z;
}

__global__ void drift(float4* pos, const float4* vel, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pos[i].x += dt * vel[i].x;
    pos[i].y += dt * vel[i].y;
    pos[i].z += dt * vel[i].z;
}

struct DeviceArray {
    float4* ptr = nullptr;
    explicit DeviceArray(std::size_t n) { check(cudaMalloc(&ptr, n * sizeof(float4)), "cudaMalloc"); }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

int blocks_for(int n) { return (n + kTile - 1) / kTile; }

void launch(Kernel kernel, const float4* pos, float4* acc, int n, float eps2) {
    if (kernel == Kernel::Naive)
        accel_naive<<<blocks_for(n), kTile>>>(pos, acc, n, eps2);
    else
        accel_tiled<<<blocks_for(n), kTile>>>(pos, acc, n, eps2);
    check(cudaGetLastError(), "kernel launch");
}

}  // namespace

static_assert(sizeof(Body4) == sizeof(float4), "Body4 must match float4");

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<Body4> accelerations(const std::vector<Body4>& pos, float eps, Kernel kernel) {
    const int n = static_cast<int>(pos.size());
    DeviceArray d_pos(n), d_acc(n);
    check(cudaMemcpy(d_pos.ptr, pos.data(), n * sizeof(float4), cudaMemcpyHostToDevice), "copy in");
    launch(kernel, d_pos.ptr, d_acc.ptr, n, eps * eps);
    std::vector<Body4> acc(n);
    check(cudaMemcpy(acc.data(), d_acc.ptr, n * sizeof(float4), cudaMemcpyDeviceToHost), "copy out");
    return acc;
}

double time_accelerations(const std::vector<Body4>& pos, float eps, Kernel kernel, int reps) {
    const int n = static_cast<int>(pos.size());
    DeviceArray d_pos(n), d_acc(n);
    check(cudaMemcpy(d_pos.ptr, pos.data(), n * sizeof(float4), cudaMemcpyHostToDevice), "copy in");
    launch(kernel, d_pos.ptr, d_acc.ptr, n, eps * eps);  // warm-up
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    check(cudaEventRecord(start), "record");
    for (int r = 0; r < reps; ++r) launch(kernel, d_pos.ptr, d_acc.ptr, n, eps * eps);
    check(cudaEventRecord(stop), "record");
    check(cudaEventSynchronize(stop), "sync");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

void integrate(System& s, float dt, int steps, float eps) {
    const int n = static_cast<int>(s.size());
    DeviceArray d_pos(n), d_vel(n), d_acc(n);
    check(cudaMemcpy(d_pos.ptr, s.pos.data(), n * sizeof(float4), cudaMemcpyHostToDevice), "copy in");
    check(cudaMemcpy(d_vel.ptr, s.vel.data(), n * sizeof(float4), cudaMemcpyHostToDevice), "copy in");
    const float eps2 = eps * eps;
    launch(Kernel::Tiled, d_pos.ptr, d_acc.ptr, n, eps2);
    for (int k = 0; k < steps; ++k) {
        kick<<<blocks_for(n), kTile>>>(d_vel.ptr, d_acc.ptr, n, 0.5f * dt);
        drift<<<blocks_for(n), kTile>>>(d_pos.ptr, d_vel.ptr, n, dt);
        launch(Kernel::Tiled, d_pos.ptr, d_acc.ptr, n, eps2);
        kick<<<blocks_for(n), kTile>>>(d_vel.ptr, d_acc.ptr, n, 0.5f * dt);
    }
    check(cudaGetLastError(), "integrate");
    check(cudaMemcpy(s.pos.data(), d_pos.ptr, n * sizeof(float4), cudaMemcpyDeviceToHost), "copy out");
    check(cudaMemcpy(s.vel.data(), d_vel.ptr, n * sizeof(float4), cudaMemcpyDeviceToHost), "copy out");
}

}  // namespace nb::gpu
