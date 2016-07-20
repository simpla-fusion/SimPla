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

    size_type dims[4];
    size_type shape[4];
    size_type ghost_width[4];
    size_type offset[4];
    size_type i_lower[4];
    size_type i_upper[4];
    Real x_lower[4];
    Real x_upper[4];
    Real dx[4];
};

void spMeshCreate(spMesh **m)
{
    *m = (spMesh *) malloc(sizeof(spMesh));
    (*m)->shape[0] = 1;
    (*m)->shape[1] = 1;
    (*m)->shape[2] = 1;
    (*m)->shape[3] = 3;

    (*m)->dims[0] = 1;
    (*m)->dims[1] = 1;
    (*m)->dims[2] = 1;
    (*m)->dims[3] = 3;

    (*m)->ghost_width[0] = 0;
    (*m)->ghost_width[1] = 0;
    (*m)->ghost_width[2] = 0;

    (*m)->i_lower[0] = 0;
    (*m)->i_lower[1] = 0;
    (*m)->i_lower[2] = 0;
    (*m)->i_lower[3] = 0;


    (*m)->i_upper[0] = 1;
    (*m)->i_upper[1] = 1;
    (*m)->i_upper[2] = 1;
    (*m)->i_upper[3] = 3;


    (*m)->x_lower[0] = 0;
    (*m)->x_lower[1] = 0;
    (*m)->x_lower[2] = 0;

    (*m)->x_upper[0] = 1;
    (*m)->x_upper[1] = 1;
    (*m)->x_upper[2] = 1;
}
void spMeshDestroy(spMesh **ctx)
{
    free(*ctx);
    *ctx = NULL;
}

void spMeshDeploy(spMesh *self)
{

    self->ndims = 3;
    for (int i = 0; i < 3; ++i)
    {
        if (self->dims[i] <= 1)
        {
            self->ghost_width[i] = 0;
            self->dims[i] = 1;
        }

        self->shape[i] = self->dims[i] + self->ghost_width[i] * 2;

        self->i_lower[i] = self->ghost_width[i];

        self->i_upper[i] = self->dims[i] + self->ghost_width[i];

    }
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
    return self->dims[0] * self->dims[1] * self->dims[2] * ((iform == 0 || iform == 3) ? 1 : 3);
}
void spMeshPoint(spMesh const *m, MeshEntityId id, Real *res)
{
    res[0] = m->x_lower[0] + m->dx[0] * (id.x - (m->i_lower[0] << 1)) * 0.5;
    res[1] = m->x_lower[1] + m->dx[1] * (id.y - (m->i_lower[1] << 1)) * 0.5;
    res[2] = m->x_lower[2] + m->dx[2] * (id.z - (m->i_lower[2] << 1)) * 0.5;
};

void spMeshSetDims(spMesh *m, size_type const *dims) { for (int i = 0; i < 3; ++i) { m->dims[i] = dims[i]; }};

size_type const *spMeshGetDims(spMesh const *m) { return m->dims; };

size_type const *spMeshGetShape(spMesh const *m) { return m->shape; };

void spMeshSetGhostWidth(spMesh *m, size_type const *gw) { for (int i = 0; i < 3; ++i) { m->ghost_width[i] = gw[i]; }};

size_type const *spMeshGetGhostWidth(spMesh const *m) { return m->ghost_width; };

void spMeshSetBox(spMesh *m, Real const *lower, Real const *upper)
{
    for (int i = 0; i < 3; ++i)
    {
        m->x_lower[i] = lower[i];
        m->x_upper[i] = upper[i];
    }
};

void spMeshGetBox(spMesh const *m, Real *lower, Real *upper)
{

    for (int i = 0; i < 3; ++i)
    {
        lower[i] = m->x_lower[i];
        upper[i] = m->x_upper[i];
    }
};

int spMeshGetDomain(spMesh const *m, int tag, size_type *lower, size_type *upper, int *o)
{

    int is_valid = 1;

    int offset[3];

    offset[0] = (tag % 3) - 1;
    offset[1] = (tag / 3) % 3 - 1;
    offset[2] = (tag / 9) % 3 - 1;

    for (int i = 0; i < 3; ++i)
    {
        switch (offset[i])
        {
            case -1:
                lower[i] = 0;
                upper[i] = m->i_lower[i];
                if (m->ghost_width[i] == 0) { is_valid = 0; }
                break;
            case 1:
                lower[i] = m->i_upper[i];
                upper[i] = m->shape[i];
                if (m->ghost_width[i] == 0) { is_valid = 0; }
                break;
            default: //0
                lower[i] = m->i_lower[i];
                upper[i] = m->i_upper[i];
                break;
        }
    }

    if (o != NULL) { for (int i = 0; i < 3; ++i) { o[i] = offset[i]; }}

    return is_valid;
};


size_type spMeshHash(spMesh const *m, MeshEntityId id, int iform)
{
    return 0;
};


void spMeshWrite(const spMesh *ctx, const char *name, int flag);

void spMeshRead(spMesh *ctx, char const name[], int flag);