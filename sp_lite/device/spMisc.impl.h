//
// Created by salmon on 16-8-15.
//

#ifndef SIMPLA_SPMISC_IMPL_H
#define SIMPLA_SPMISC_IMPL_H


#include "../spMisc.h"
#include "sp_device.h"
#include <math.h>


SP_DEVICE_DECLARE_KERNEL(spFieldAssignValueSinKernel_g, Real *data, dim3 strides, Real3 k_dx, Real3 alpha0, Real amp)
{
    size_type x = threadIdx.x + blockIdx.x * blockDim.x;
    size_type y = threadIdx.y + blockIdx.y * blockDim.y;
    size_type z = threadIdx.z + blockIdx.z * blockDim.z;

    size_type s = x * strides.x + y * strides.y + z * strides.z;

    data[s] = amp * (Real) (cos(k_dx.x * (x) + alpha0.x) * cos(k_dx.y * (y) + alpha0.y) * cos(k_dx.z * (z) + alpha0.z));
}


#define HALFPI (3.1415926*0.5)

int spFieldAssignValueSin(spField *f, Real const *k, Real const *amp)
{

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) f);
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) f);
    int ndims = spMeshGetNDims(m);
    int num_of_sub = spFieldNumberOfSub(f);

    Real *data[num_of_sub];
    size_type dims[4], start[4], count[4];

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, dims, start, count));

    size_type strides[4];

    Real x0[3];
    spMeshGetOrigin(m, x0);
    Real dx[3];
    spMeshGetDx(m, dx);

    SP_CALL(spMeshGetStrides(m, strides));

    size_type offset = start[0] * strides[0] + start[1] * strides[1] + start[2] * strides[2];

    Real k_dx[3] = {k[0] * dx[0], k[1] * dx[1], k[2] * dx[2]};

    Real alpha0[9];

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

    SP_CALL(spFieldSubArray(f, (void **) data));

    dim3 gridDim = sizeType2Dim3(count);

    dim3 blockDim = {1, 1, 1};

    for (int i = 0; i < num_of_sub; ++i)
    {
        dim3 t_strides = sizeType2Dim3(strides);
        Real3 t_k_dx = real2Real3(k_dx);
        Real3 t_alpha0 = real2Real3(alpha0 + i * 3);

        SP_DEVICE_CALL_KERNEL(spFieldAssignValueSinKernel_g,
                              sizeType2Dim3(count), blockDim,
                              data[i] + offset,
                              t_strides, t_k_dx, t_alpha0,
                              amp[i]
        );
    }

    SP_CALL(spFieldSync(f));

    return SP_SUCCESS;
};


#endif //SIMPLA_SPMISC_IMPL_H
