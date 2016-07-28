//
// Created by salmon on 16-7-27.
//
#include "sp_lite_def.h"
#include "spMisc.h"
#include "spParallel.h"
#include "spField.h"
#include "spMesh.h"
#define HALFPI (3.1415926*0.5)

int spFieldAssignValueSin(spField *f, Real const *k, Real const *amp)
{

    spMesh const *m = spMeshAttrMesh((spMeshAttr const *) f);
    int iform = spMeshAttrForm((spMeshAttr const *) f);
    int ndims = spMeshNDims(m);
    int num_of_sub = spFieldNumberOfSub(f);

    Real *data[num_of_sub];
    size_type dims[4], start[4], count[4];

    SP_CALL(spMeshLocalDomain(m, SP_DOMAIN_CENTER, dims, start, count));

    size_type strides[4];

    Real const *x0 = spMeshGetLocalOrigin(m);
    Real const *dx = spMeshGetDx(m);
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

    size_type num_of_threads = 1;

    for (int i = 0; i < num_of_sub; ++i)
    {
        spFieldAssignValueSinKernel(count,
                                    &num_of_threads,
                                    data[i] + offset,
                                    (strides),
                                    (k_dx),
                                    (alpha0 + i * 3),
                                    amp[i]);
    }

    SP_CALL(spFieldSync(f));

    return SP_SUCCESS;
};
