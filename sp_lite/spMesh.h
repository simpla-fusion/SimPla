/*
 * spMesh.h
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifndef SPMESH_H_
#define SPMESH_H_

#include "sp_lite_def.h"
#include "spObject.h"

enum { VERTEX = 0, EDGE = 1, FACE = 2, VOLUME = 3 };

struct spMesh_s;

typedef struct spMesh_s spMesh;

#define SP_MESH_ATTR_HEAD  SP_OBJECT_HEAD const spMesh *m; int iform;

typedef struct spMeshAttr_s
{

    SP_MESH_ATTR_HEAD
    byte_type __others[];
} spMeshAttr;

MeshEntityId spMeshEntityIdFromArray(size_type const *s);

MeshEntityId spMeshEntityIdShift(MeshEntityId id, ptrdiff_t const *s);

int spMeshAttrCreate(spMeshAttr **f, size_type size, spMesh const *mesh, int iform);

int spMeshAttrDestroy(spMeshAttr **f);

spMesh const *spMeshAttrMesh(spMeshAttr const *f);

int spMeshAttrForm(spMeshAttr const *f);

int spMeshCreate(spMesh **ctx);

int spMeshDestroy(spMesh **ctx);

int spMeshDeploy(spMesh *self);

int spMeshNDims(spMesh const *m);

Real spMeshCFLDtv(spMesh const *m, Real const *speed);

Real spMeshCFLDt(spMesh const *m, Real const speed);

int spMeshSetDims(spMesh *m, size_type const *);

size_type const *spMeshGetDims(spMesh const *m);

int spMeshSetGhostWidth(spMesh *m, size_type const *);

size_type const *spMeshGetGhostWidth(spMesh const *m);

int spMeshSetBox(spMesh *m, Real const *lower, Real const *upper);

Real const *spMeshGetLocalOrigin(spMesh const *m);

Real const *spMeshGetGlobalOrigin(spMesh const *m);

Real const *spMeshGetDx(spMesh const *m);

Real const *spMeshGetInvDx(spMesh const *m);

int spMeshGetLocalBox(spMesh const *m, int tag, Real *lower, Real *upper);

int spMeshGetGlobalBox(spMesh const *m, Real *lower, Real *upper);

#define SP_DOMAIN_CENTER 13
#define SP_DOMAIN_ALL 0xFF


int spMeshGetStrides(spMesh const *m, size_type *res);

int spMeshLocalDomain(spMesh const *m, int tag, size_type *dims, size_type *start, size_type *count);

int spMeshLocalDomain2(spMesh const *m, int tag, size_type *min, size_type *max, size_type *stride);

int spMeshGlobalOffset(spMesh const *m, size_type *dims, ptrdiff_t *offset);

int spMeshArrayShape(spMesh const *m,
                     int domain_tag,
                     int attr_ndims,
                     size_type const *attr_dims,
                     int *array_ndims,
                     int *start_mesh_dim,
                     size_type *g_dims,
                     size_type *g_start,
                     size_type *l_dims,
                     size_type *l_start,
                     size_type *l_count,
                     int is_soa);

size_type spMeshNumberOfEntity(spMesh const *, int domain_tag, int iform);

size_type spMeshHash(spMesh const *, MeshEntityId, int iform);

void spMeshPoint(spMesh const *, MeshEntityId id, Real *);

int spMeshWrite(const spMesh *ctx, const char *name, int flag);

int spMeshRead(spMesh *ctx, const char *name, int flag);

#endif /* SPMESH_H_ */
