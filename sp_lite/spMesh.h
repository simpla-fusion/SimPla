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

void spMeshSetDims(spMesh *m, size_type const *);

size_type const *spMeshGetDims(spMesh const *m);

size_type const *spMeshGetShape(spMesh const *m);

void spMeshSetGhostWidth(spMesh *m, size_type const *);

size_type const *spMeshGetGhostWidth(spMesh const *m);

void spMeshSetBox(spMesh *m, Real const *lower, Real const *upper);

void spMeshGetBox(spMesh const *m, Real *lower, Real *upper);

#define SP_DOMAIN_CENTER 13
#define SP_DOMAIN_ALL 0xFF

void spMeshGetDx(spMesh const *m, Real *dx);

int spMeshDomain(spMesh const *m, int tag, size_type *shape, size_type *lower, size_type *upper, int *offset);

size_type spMeshNumberOfEntity(spMesh const *, int domain_tag, int iform);

size_type spMeshHash(spMesh const *, MeshEntityId, int iform);

void spMeshPoint(spMesh const *, MeshEntityId id, Real *);

void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);

#endif /* SPMESH_H_ */
