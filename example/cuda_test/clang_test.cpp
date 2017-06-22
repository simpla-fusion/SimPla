//
// Created by salmon on 17-6-22.
//

#include <stdio.h>
#include <cmath>

//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <device_launch_parameters.h>
//#include <driver_types.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/nTuple.h>

using namespace simpla;

template <typename T>
__device__ auto foo(T const &a) {
    return a * 2.0;
}

__global__ void saxpy(int n, nTuple<Real, 3> const *a, nTuple<Real, 3> const *b, nTuple<Real, 3> *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = a[i] + b[i];
}

int main(void) {
    int N = 1 << 20;
    nTuple<Real, 3> *a;
    nTuple<Real, 3> *b;
    nTuple<Real, 3> *y;
    nTuple<Real, 3> c, d;

    d = 1.0;

    c = d * 2.0;

    //    std::cout << c << std::endl;

    cudaMalloc(&a, N * sizeof(nTuple<Real, 3>));
    cudaMalloc(&b, N * sizeof(nTuple<Real, 3>));
    cudaMalloc(&y, N * sizeof(nTuple<Real, 3>));

    //    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 255) / 256, 256>>>(N, a, b, y);

    //    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    //    float maxError = 0.0f;
    //    for (int i = 0; i < N; i++) maxError = std::fmax(maxError, abs(y[i] - 4.0f));
    //    printf("Max error: %f\n", maxError);

    cudaFree(&a);
    cudaFree(&b);
    cudaFree(&c);
}