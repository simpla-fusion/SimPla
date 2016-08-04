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

#define SP_MESH_ATTR_HEAD  SP_OBJECT_HEAD const spMesh *m; int iform;

typedef struct spMeshAttribute_s
{

    SP_MESH_ATTR_HEAD
    byte_type __others[];
} spMeshAttribute;

MeshEntityId spMeshEntityIdFromArray(size_type const *s);

MeshEntityId spMeshEntityIdShift(MeshEntityId id, ptrdiff_t const *s);

int spMeshAttributeCreate(spMeshAttribute **f, size_type size, spMesh const *mesh, int iform);

int spMeshAttributeDestroy(spMeshAttribute **f);

spMesh const *spMeshAttributeGetMesh(spMeshAttribute const *f);

int spMeshAttributeGetForm(spMeshAttribute const *f);

int spMeshCreate(spMesh **ctx);

int spMeshDestroy(spMesh **ctx);

int spMeshDeploy(spMesh *self);

/** Topology Begin*/

int spMeshGetNDims(spMesh const *m);

Real spMeshCFLDtv(spMesh const *m, Real const *speed);

Real spMeshCFLDt(spMesh const *m, Real const speed);

int spMeshSetDims(spMesh *m, size_type const *);

int spMeshGetDims(spMesh const *m, size_type *);

int spMeshSetGhostWidth(spMesh *m, size_type const *);

int spMeshGetGhostWidth(spMesh const *m, size_type *);

int spMeshGetStrides(spMesh const *m, size_type *res);

size_type spMeshGetNumberOfEntities(spMesh const *, int domain_tag, int iform);

int spMeshGetDomain(spMesh const *m, int tag, size_type *dims, size_type *start, size_type *count);

int spMeshGetArrayShape(spMesh const *m, int tag, size_type *min, size_type *max, size_type *stride);

int spMeshGetGlobalOffset(spMesh const *m, size_type *dims, ptrdiff_t *offset);

int spMeshGetGlobalArrayShape(spMesh const *m,
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
int spMeshSetBox(spMesh *m, Real const *lower, Real const *upper);

int spMeshGetBox(spMesh const *m, int tag, Real *lower, Real *upper);

int spMeshGetOrigin(spMesh const *m, Real *origin);

int spMeshGetDx(spMesh const *m, Real *);

int spMeshGetInvDx(spMesh const *m, Real *);

int spMeshGetGlobalOrigin(spMesh const *m, Real *origin);

size_type spMeshHash(spMesh const *, MeshEntityId, int iform);

void spMeshPoint(spMesh const *, MeshEntityId id, Real *);

/** Geometry End */

int spMeshWrite(const spMesh *ctx, const char *name, int flag);

int spMeshRead(spMesh *ctx, const char *name, int flag);

#endif /* SPMESH_H_ */
