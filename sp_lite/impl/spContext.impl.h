//
// Created by salmon on 16-8-19.
//

#ifndef SIMPLA_SPSIMULATION_H
#define SIMPLA_SPSIMULATION_H
#include "../sp_lite_def.h"
#include "sp_device.h"
#include "../spMesh.h"

typedef struct
{
    uint3 min;
    uint3 max;
    uint3 strides;
    float3 inv_dx;

    uint3 grid_dim;
    uint3 block_dim;

} spContextConstantDevice;

extern __constant__ spContextConstantDevice sp_ctx_d;

typedef struct spContext_s
{
    spMesh *m;

} spContext;

extern spContext sp_ctx;

int spContextCreate(spContext *);

#endif //SIMPLA_SPSIMULATION_H
