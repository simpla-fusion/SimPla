/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_

#include "sp_lite_def.h"
#include "spParallel.h"
#include "spObject.h"

enum
{
    VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3
};
union MeshEntityId_u
{
    struct
    {
        int16_t w, z, y, x;
    };
    int64_t v;

};

typedef union MeshEntityId_u MeshEntityId;

struct spMesh_s
{

    int ndims;

    dim3 dims;
    dim3 offset;
    dim3 i_lower;
    dim3 i_upper;
    Real3 x_lower;
    Real3 dx;

    size_type number_of_idx;
    size_type *cell_idx;

    size_type number_of_shared_blocks;
    dim3 *shared_blocks;
    dim3 private_block;
    dim3 threadsPerBlock;

//	spDistributedObject *dist_obj;

};

typedef struct spMesh_s spMesh;

void spMeshCreate(spMesh **ctx);

void spMeshDestroy(spMesh **ctx);

void spMeshDeploy(spMesh *self);

void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);

size_t spMeshGetNumberOfEntity(spMesh const *, int iform);

Real3 spMeshPoint(spMesh const *, MeshEntityId id);

#endif /* SPMESH_H_ */
