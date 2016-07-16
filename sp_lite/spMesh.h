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

enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3 };

union MeshEntityId_u
{
    struct { int16_t w, z, y, x; };
    int64_t v;
};

typedef union MeshEntityId_u MeshEntityId;

struct spMesh_s;

typedef struct spMesh_s spMesh;

void spMeshCreate(spMesh **ctx);

void spMeshDestroy(spMesh **ctx);

void spMeshDeploy(spMesh *self);

void spMeshSetDims(spMesh *m, dim3);

dim3 spMeshGetDims(spMesh const *m);

dim3 spMeshGetShape(spMesh const *m);

void spMeshSetGhostWidth(spMesh *m, dim3);

dim3 spMeshGetGhostWidth(spMesh const *m, dim3 *);

void spMeshSetBox(spMesh *m, Real3 lower, Real3 upper);

void spMeshGetBox(spMesh const *m, Real3 *lower, Real3 *upper);

void spMeshGetDomain(spMesh const *m, int tag, dim3 *lower, dim3 *upper, dim3 *offset);

size_type spMeshGetNumberOfEntity(spMesh const *, int iform);

size_type spMeshHash(spMesh const *, MeshEntityId, int iform);

Real3 spMeshPoint(spMesh const *, MeshEntityId id);

void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);

#endif /* SPMESH_H_ */
