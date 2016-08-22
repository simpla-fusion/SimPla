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
enum { SP_DOMAIN_CENTER = 13, SP_DOMAIN_ALL = 0xFF };
struct spMesh_s;

typedef struct spMesh_s spMesh;

typedef struct spMesh_s *spMesh_t;

typedef struct spMesh_s const *spMesh_const_t;

#define SP_MESH_ATTR_HEAD  SP_OBJECT_HEAD const spMesh *m; int iform;

typedef struct spMeshAttribute_s
{
    SP_MESH_ATTR_HEAD
    byte_type __others[];
} spMeshAttribute;

typedef struct spMeshAttribute_s *spMeshAttribute_t;

typedef struct spMeshAttribute_s const *spMeshAttribute_const_t;

MeshEntityId spMeshEntityIdFromArray(size_type const *s);

MeshEntityId spMeshEntityIdShift(MeshEntityId id, ptrdiff_t const *s);

int spMeshAttributeCreate(spMeshAttribute_t *f, size_type size, spMesh_const_t mesh, int iform);

int spMeshAttributeDestroy(spMeshAttribute_t *f);

spMesh_const_t spMeshAttributeGetMesh(spMeshAttribute_const_t f);

int spMeshAttributeGetForm(spMeshAttribute_const_t f);

int spMeshCreate(spMesh_t *ctx);

int spMeshDestroy(spMesh_t *ctx);

int spMeshDeploy(spMesh_t self);

/** Topology Begin*/

int spMeshGetNDims(spMesh_const_t m);

Real spMeshCFLDtv(spMesh_const_t m, Real const *speed);

Real spMeshCFLDt(spMesh_const_t m, Real const speed);

int spMeshSetDims(spMesh_t m, size_type const *);

int spMeshGetDims(spMesh_const_t m, size_type *);

int spMeshSetGhostWidth(spMesh_t m, size_type const *);

int spMeshGetGhostWidth(spMesh_const_t m, size_type *);

int spMeshGetStrides(spMesh_const_t m, size_type *res);

size_type spMeshGetNumberOfEntities(spMesh_const_t, int domain_tag, int iform);

int spMeshGetDomain(spMesh_const_t m, int tag, size_type *dims, size_type *start, size_type *count);

int spMeshGetArrayShape(spMesh_const_t m, int tag, size_type *min, size_type *max, size_type *stride);

int spMeshGetGlobalOffset(spMesh_const_t m, size_type *dims, ptrdiff_t *offset);

int spMeshGetGlobalArrayShape(spMesh_const_t m,
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
/** Topology End*/

/** Geometry Begin*/
int spMeshSetBox(spMesh_t m, Real const *lower, Real const *upper);

int spMeshGetBox(spMesh_const_t m, int tag, Real *lower, Real *upper);

int spMeshGetOrigin(spMesh_const_t m, Real *origin);

int spMeshGetDx(spMesh_const_t m, Real *);

int spMeshGetInvDx(spMesh_const_t m, Real *);

int spMeshGetGlobalOrigin(spMesh_const_t m, Real *origin);

size_type spMeshHash(spMesh_const_t, MeshEntityId, int iform);

void spMeshPoint(spMesh_const_t, MeshEntityId id, Real *);

__inline__ size_type spMeshSFC(size_type const *d, size_type const *strides)
{
    return d[0] * strides[0] + d[1] * strides[1] + d[2] * strides[2];
}

/** Geometry End */

int spMeshWrite(spMesh_const_t ctx, const char *name);

int spMeshRead(spMesh_t ctx, const char *name);

#endif /* SPMESH_H_ */
