//
// Created by salmon on 17-6-22.
//

#include <host_defines.h>
#include <simpla/utilities/ExpressionTemplate.h>
#include <simpla/utilities/nTuple.ext.h>
#include <simpla/utilities/nTuple.h>

using namespace simpla;

template <typename T>
__device__ auto foo(T const &a) {
    return a * 2.0;
}
template <typename T>
__global__ void saxpy(int n, T *a, T *b, T *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = i;
    b[i] = 2 * i;
    y[i] = a[i] + b[i];
}

int main(void) {
    int N = 1 << 10;
    typedef nTuple<Real, 3> value_type;
    value_type *a;
    value_type *b;
    value_type *x, *y;
    value_type c, d;

    x = reinterpret_cast<value_type *>(malloc(N * sizeof(value_type)));
    for (int i = 0; i < N; ++i) { x[i] = 0; }
    //    for (int i = 0; i < 100; i++) { std::cout << i << "\t=" << x[i] << std::endl; }

    d = 1.0;

    c = d * 2.0;

    //    std::cout << c << std::endl;

    cudaMalloc(&a, N * sizeof(value_type));
    cudaMalloc(&b, N * sizeof(value_type));
    cudaMalloc(&y, N * sizeof(value_type));

    //    cudaMemcpy(y, x, N * sizeof(value_type), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 32) / 32, 32>>>(N, a, b, y);

    cudaMemcpy(x, y, N * sizeof(value_type), cudaMemcpyDefault);
    //    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDefault);

    //    float maxError = 0.0f;
    for (int i = 0; i < 100; i++) { std::cout << i << "\t=" << x[i] << std::endl; }
    //    printf("Max error: %f\n", maxError);

    cudaFree(&a);
    cudaFree(&b);
    cudaFree(&c);
    free(x);
}