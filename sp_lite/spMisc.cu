//
// Created by salmon on 16-7-27.
//

extern "C" {
#include <cmath>
#include "spMisc.h"
#include "sp_lite_def.h"
#include "spParallel.h"
#include "spParallelCUDA.h"
#include "spField.h"
#include "spMesh.h"
}
__global__
void spFieldAssignValueSinKernel(Real *data, dim3 strides, Real3 k_dx, Real3 alpha0, Real amp)
{
    size_type x = threadIdx.x + blockIdx.x * blockDim.x;
    size_type y = threadIdx.y + blockIdx.y * blockDim.y;
    size_type z = threadIdx.z + blockIdx.z * blockDim.z;

    size_type s = x * strides.x + y * strides.y + z * strides.z;

    data[s] = amp *
        sin(k_dx.x * Real(x) + alpha0.x) *
        sin(k_dx.y * Real(y) + alpha0.y) *
        sin(k_dx.z * Real(z) + alpha0.z);

}
#define HALFPI (3.1415926*0.5)
int spFieldAssignValueSin(spField *f, Real const *k, Real const *amp)
{

    spMesh const *m = spMeshAttrMesh((spMeshAttr const *) f);
    int iform = spMeshAttrForm((spMeshAttr const *) f);
    int ndims = spMeshNDims(m);
    int num_of_sub = spFieldNumberOfSub(f);

    Real *data[num_of_sub];

    dim3 block_dim, thread_dim;

    size_type dims[4], start[4], count[4];

    SP_CHECK_RETURN(spMeshLocalDomain(m, SP_DOMAIN_CENTER, dims, start, count));


    size_type strides[4];

    Real const *x0 = spMeshGetLocalOrigin(m);

    Real const *dx = spMeshGetDx(m);

    spMeshGetStrides(m, strides);

    Real k_dx[3] = {k[0] * dx[0], k[1] * dx[1], k[2] * dx[2]};

    Real alpha0[9];
    CHECK_INT(dims[0])
    CHECK_INT(dims[1])
    CHECK_INT(dims[2])
    switch (iform)
    {
        case EDGE:
            alpha0[0] = dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5));
            alpha0[1] = dims[1] == 1 ? HALFPI : (k[1] * x0[1]);
            alpha0[2] = dims[2] == 1 ? HALFPI : (k[2] * x0[2]);
            alpha0[3] = dims[0] == 1 ? HALFPI : (k[0] * x0[0]);
            alpha0[4] = dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5));
            alpha0[5] = dims[2] == 1 ? HALFPI : (k[2] * x0[2]);
            alpha0[6] = dims[0] == 1 ? HALFPI : (k[0] * x0[0]);
            alpha0[7] = dims[1] == 1 ? HALFPI : (k[1] * x0[1]);
            alpha0[8] = dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5));
            break;
        case FACE:
            alpha0[0] = dims[0] == 1 ? HALFPI : (k[0] * x0[0]);
            alpha0[1] = dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5));
            alpha0[2] = dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5));
            alpha0[3] = dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5));
            alpha0[4] = dims[1] == 1 ? HALFPI : (k[1] * x0[1]);
            alpha0[5] = dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5));
            alpha0[6] = dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5));
            alpha0[7] = dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5));
            alpha0[8] = dims[2] == 1 ? HALFPI : (k[2] * x0[2]);
            break;
        case VOLUME:

            alpha0[0] = dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5));
            alpha0[1] = dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5));
            alpha0[2] = dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5));
            alpha0[3] = dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5));
            alpha0[4] = dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5));
            alpha0[5] = dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5));
            alpha0[6] = dims[0] == 1 ? HALFPI : (k[0] * (x0[0] + dx[0] * 0.5));
            alpha0[7] = dims[1] == 1 ? HALFPI : (k[1] * (x0[1] + dx[1] * 0.5));
            alpha0[8] = dims[2] == 1 ? HALFPI : (k[2] * (x0[2] + dx[2] * 0.5));
            break;

        case VERTEX:
        default:
            alpha0[0] = dims[0] == 1 ? HALFPI : (k[0] * x0[0]);
            alpha0[1] = dims[1] == 1 ? HALFPI : (k[1] * x0[1]);
            alpha0[2] = dims[2] == 1 ? HALFPI : (k[2] * x0[2]);
            alpha0[3] = dims[0] == 1 ? HALFPI : (k[0] * x0[0]);
            alpha0[4] = dims[1] == 1 ? HALFPI : (k[1] * x0[1]);
            alpha0[5] = dims[2] == 1 ? HALFPI : (k[2] * x0[2]);
            alpha0[6] = dims[0] == 1 ? HALFPI : (k[0] * x0[0]);
            alpha0[7] = dims[1] == 1 ? HALFPI : (k[1] * x0[1]);
            alpha0[8] = dims[2] == 1 ? HALFPI : (k[2] * x0[2]);
    };
    SP_CHECK_RETURN(spFieldSubArray(f, SP_DOMAIN_CENTER, (void **) data, NULL));


    for (int i = 0; i < num_of_sub; ++i)
    {
        LOAD_KERNEL(spFieldAssignValueSinKernel, sizeType2Dim3(count), 1,
                    data[i], sizeType2Dim3(strides), real2Real3(k_dx), real2Real3(alpha0 + i * 3), amp[i]);
    }

    return SP_SUCCESS;
};
