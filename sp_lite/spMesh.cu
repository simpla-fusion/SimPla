/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <stdlib.h>
#include <assert.h>
#include "sp_lite_def.h"
#include "spMesh.h"
#include "spParallel.h"


struct spMesh_s
{

    int ndims;

    dim3 dims;
    dim3 shape;
    dim3 ghost_width;

    dim3 offset;
    dim3 i_lower;
    dim3 i_upper;
    Real3 x_lower;
    Real3 x_upper;
    Real3 dx;
};

void spMeshCreate(spMesh **m)
{
    *m = (spMesh *) malloc(sizeof(spMesh));
    (*m)->shape.x = 1;
    (*m)->shape.y = 1;
    (*m)->shape.z = 1;

    (*m)->dims.x = 1;
    (*m)->dims.y = 1;
    (*m)->dims.z = 1;

    (*m)->ghost_width.x = 0;
    (*m)->ghost_width.y = 0;
    (*m)->ghost_width.z = 0;

    (*m)->i_lower.x = 0;
    (*m)->i_lower.y = 0;
    (*m)->i_lower.z = 0;


    (*m)->i_upper.x = 1;
    (*m)->i_upper.y = 1;
    (*m)->i_upper.z = 1;


    (*m)->x_lower.x = 0;
    (*m)->x_lower.y = 0;
    (*m)->x_lower.z = 0;

    (*m)->x_upper.x = 1;
    (*m)->x_upper.y = 1;
    (*m)->x_upper.z = 1;
}
void spMeshDestroy(spMesh **ctx)
{
    free(*ctx);
    *ctx = NULL;
}

void spMeshDeploy(spMesh *self)
{

    self->ndims = 3;
    self->shape.x = self->ghost_width.x * 2 + self->dims.x;
    self->shape.y = self->ghost_width.y * 2 + self->dims.y;
    self->shape.z = self->ghost_width.z * 2 + self->dims.z;

    self->i_lower.x = self->ghost_width.x;
    self->i_lower.y = self->ghost_width.y;
    self->i_lower.z = self->ghost_width.z;

    self->i_upper.x = self->dims.x + self->ghost_width.x;
    self->i_upper.y = self->dims.y + self->ghost_width.y;
    self->i_upper.z = self->dims.z + self->ghost_width.z;


    /**          -1
     *
     *    -1     0    1
     *
     *           1
     */
    /**
     *\verbatim
     *                ^y
     *               /
     *        z     /
     *        ^    /
     *    PIXEL0 110-------------111 VOXEL
     *        |  /|              /|
     *        | / |             / |
     *        |/  |    PIXEL1  /  |
     * EDGE2 100--|----------101  |
     *        | m |           |   |
     *        |  010----------|--011 PIXEL2
     *        |  / EDGE1      |  /
     *        | /             | /
     *        |/              |/
     *       000-------------001---> x
     *                       EDGE0
     *
     *\endverbatim
     */
//	int3 neighbour_offset[27];
//	int neighbour_flag[27];
//	int count = 0;
//	for (int i = -1; i <= 1; ++i)
//		for (int j = -1; j <= 1; ++j)
//			for (int k = -1; k <= 1; ++k)
//			{
//				neighbour_offset[count].x = i;
//				neighbour_offset[count].y = j;
//				neighbour_offset[count].z = k;
//				neighbour_flag[count] = (i + 1) | ((j + 1) << 2) | ((k + 1) << 4);
//				++count;
//			}
//	assert(count == 27);
//	spParallelMemcpyToSymbol(SP_NEIGHBOUR_OFFSET, neighbour_offset, sizeof(neighbour_offset));
//	spParallelMemcpyToSymbol(SP_NEIGHBOUR_OFFSET_flag, neighbour_flag, sizeof(neighbour_flag));
}

size_type spMeshGetNumberOfEntity(spMesh const *self, int iform)
{
    return self->dims.x * self->dims.y * self->dims.z * ((iform == 0 || iform == 3) ? 1 : 3);
}
Real3 spMeshPoint(spMesh const *m, MeshEntityId id)
{
    Real3 res;
    res.x = m->x_lower.x + m->dx.x * (id.x - (m->i_lower.x << 1)) * 0.5;
    res.y = m->x_lower.y + m->dx.y * (id.y - (m->i_lower.y << 1)) * 0.5;
    res.z = m->x_lower.z + m->dx.z * (id.z - (m->i_lower.z << 1)) * 0.5;
    return res;
};

void spMeshSetDims(spMesh *m, dim3 dims) { m->dims = dims; };

dim3 spMeshGetDims(spMesh const *m) { return m->dims; };

dim3 spMeshGetShape(spMesh const *m) { return m->shape; };

void spMeshSetGhostWidth(spMesh *m, dim3 gw) { m->ghost_width = gw; };

dim3 spMeshGetGhostWidth(spMesh const *m) { return m->ghost_width; };

void spMeshSetBox(spMesh *m, Real3 lower, Real3 upper)
{
    m->x_lower = lower;
    m->x_upper = upper;
};

void spMeshGetBox(spMesh const *m, Real3 *lower, Real3 *upper)
{
    *lower = m->x_lower;
    *upper = m->x_upper;
};

void spMeshGetDomain(spMesh const *m, int tag, dim3 *lower, dim3 *upper, dim3 *offset)
{
    *lower = m->i_lower;
    *upper = m->i_upper;
    if (offset != NULL)
    {
        offset->x = 0;
        offset->y = 0;
        offset->z = 0;
    }
};


size_type spMeshHash(spMesh const *m, MeshEntityId id, int iform)
{
    return 0;
};


void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);