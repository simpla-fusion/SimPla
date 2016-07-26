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


int spMeshAttrCreate(spMeshAttr **f, size_type size, spMesh const *mesh, int iform)
{
    SP_CHECK_RETURN(spObjectCreate((spObject **) (f), size));
    (*f)->m = mesh;
    (*f)->iform = iform;
    return SP_SUCCESS;
};

int spMeshAttrDestroy(spMeshAttr **f)
{
    SP_CHECK_RETURN(spObjectDestroy((spObject **) (f)));
    return SP_SUCCESS;
};
spMesh const *spMeshAttrMesh(spMeshAttr const *f) { return f->m; }

int spMeshAttrForm(spMeshAttr const *f) { return f->iform; };

struct spMesh_s
{

    int ndims;
    size_type ghost_width[4];
    size_type local_dims[4];
    size_type local_start[4];
    size_type local_count[4];

    size_type global_dims[4];
    size_type global_start[4];

    Real x_lower[4];
    Real x_upper[4];
    Real dx[4];
};

int spMeshCreate(spMesh **m)
{
    *m = (spMesh *) malloc(sizeof(spMesh));
    (*m)->ndims = 3;

    (*m)->global_dims[0] = 1;
    (*m)->global_dims[1] = 1;
    (*m)->global_dims[2] = 1;
    (*m)->global_dims[3] = 3;

    (*m)->local_dims[0] = 1;
    (*m)->local_dims[1] = 1;
    (*m)->local_dims[2] = 1;
    (*m)->local_dims[3] = 3;

    (*m)->ghost_width[0] = 0;
    (*m)->ghost_width[1] = 0;
    (*m)->ghost_width[2] = 0;

    (*m)->local_start[0] = 0;
    (*m)->local_start[1] = 0;
    (*m)->local_start[2] = 0;
    (*m)->local_start[3] = 0;


    (*m)->local_count[0] = 1;
    (*m)->local_count[1] = 1;
    (*m)->local_count[2] = 1;
    (*m)->local_count[3] = 3;


    (*m)->x_lower[0] = 0;
    (*m)->x_lower[1] = 0;
    (*m)->x_lower[2] = 0;

    (*m)->x_upper[0] = 1;
    (*m)->x_upper[1] = 1;
    (*m)->x_upper[2] = 1;

    return SP_SUCCESS;
}

int spMeshDestroy(spMesh **ctx)
{
    free(*ctx);
    *ctx = NULL;
    return SP_SUCCESS;
}

int spMeshDeploy(spMesh *self)
{
    int mpi_topo_ndims = 0;
    int mpi_topo_dims[3];
    int mpi_topo_periods[3];
    int mpi_topo_coords[3];

    spMPITopology(&mpi_topo_ndims, mpi_topo_dims, mpi_topo_periods, mpi_topo_coords);

    self->ndims = 3;

    for (int i = 0; i < 3; ++i)
    {

        if (self->global_dims[i] <= 1)
        {
            self->ghost_width[i] = 0;

            self->global_dims[i] = 1;
        }

        if (i < mpi_topo_ndims)
        {
            self->global_start[i] = self->global_dims[i] * mpi_topo_coords[i] / mpi_topo_dims[i];

            self->local_count[i] =
                self->global_dims[i] * (mpi_topo_coords[i] + 1) / mpi_topo_dims[i] - self->global_start[i];
        }
        else
        {
            self->global_start[i] = 0;

            self->local_count[i] = self->global_dims[i];
        }

        self->local_start[i] = self->ghost_width[i];

        self->local_dims[i] = self->local_count[i] + self->ghost_width[i] * 2;

        self->dx[i] = (self->x_upper[i] - self->x_lower[i]) / self->global_dims[i];
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

    return SP_SUCCESS;

}

size_type spMeshNumberOfEntity(spMesh const *self, int tag, int iform)
{
    size_type res = (iform == 0 || iform == 3) ? 1 : 3;
    switch (tag)
    {
        case SP_DOMAIN_CENTER:
            res *= self->local_count[0] * self->local_count[1] * self->local_count[2];
        case SP_DOMAIN_ALL:
        default:
            res *= self->local_dims[0] * self->local_dims[1] * self->local_dims[2];
            break;
    }
    return res;
}

void spMeshPoint(spMesh const *m, MeshEntityId id, Real *res)
{
    res[0] = m->x_lower[0] + m->dx[0] * (id.x - (m->local_start[0] << 1)) * 0.5;
    res[1] = m->x_lower[1] + m->dx[1] * (id.y - (m->local_start[1] << 1)) * 0.5;
    res[2] = m->x_lower[2] + m->dx[2] * (id.z - (m->local_start[2] << 1)) * 0.5;
};

int spMeshNDims(spMesh const *m) { return m->ndims; };

int spMeshSetDims(spMesh *m, size_type const *dims)
{
    for (int i = 0; i < m->ndims; ++i) { m->global_dims[i] = dims[i]; }
    return SP_SUCCESS;
};
size_type const *spMeshGetDims(spMesh const *m) { return m->local_dims; }

int spMeshSetGhostWidth(spMesh *m, size_type const *gw)
{
    for (int i = 0; i < 3; ++i) { m->ghost_width[i] = gw[i]; }
    return SP_SUCCESS;
};


int spMeshSetBox(spMesh *m, Real const *lower, Real const *upper)
{
    for (int i = 0; i < 3; ++i)
    {
        m->x_lower[i] = lower[i];
        m->x_upper[i] = upper[i];
    }
    return SP_SUCCESS;

};

int spMeshGetBox(spMesh const *m, Real *lower, Real *upper)
{

    for (int i = 0; i < 3; ++i)
    {
        lower[i] = m->x_lower[i];
        upper[i] = m->x_upper[i];
    }
    return SP_SUCCESS;

};

int spMeshGetDx(spMesh const *m, Real *dx)
{
    dx[0] = m->dx[0];
    dx[1] = m->dx[1];
    dx[2] = m->dx[2];
    return SP_SUCCESS;

}

int spMeshLocalDomain(spMesh const *m, int tag, size_type *dims, size_type *start, size_type *count)
{

    int success = SP_SUCCESS;
    int offset[3] = {0, 0, 0};

    switch (tag)
    {
        case SP_DOMAIN_ALL:
            for (int i = 0; i < 3; ++i)
            {
                start[i] = 0;
                count[i] = m->local_dims[i];
            }
            break;
        case SP_DOMAIN_CENTER:
            for (int i = 0; i < 3; ++i)
            {
                start[i] = m->local_start[i];
                count[i] = m->local_count[i];
            }

            break;
        default:


            offset[0] = (tag % 3) - 1;
            offset[1] = (tag / 3) % 3 - 1;
            offset[2] = (tag / 9) % 3 - 1;

            for (int i = 0; i < 3; ++i)
            {
                switch (offset[i])
                {
                    case -1:
                        start[i] = 0;
                        count[i] = m->ghost_width[i];
                        if (m->ghost_width[i] == 0) { success = SP_FAILED; }
                        break;
                    case 1:
                        start[i] = m->local_count[i] + m->local_start[i];
                        count[i] = m->local_dims[i] - start[i];
                        if (m->ghost_width[i] == 0) { success = SP_FAILED; }
                        break;
                    default: //0
                        start[i] = m->local_start[i];
                        count[i] = m->local_count[i];
                        break;
                }
            }
    }

//    if (o != NULL) { for (int i = 0; i < 3; ++i) { o[i] = global_start[i]; }}
    if (dims != NULL) { for (int i = 0; i < 3; ++i) { dims[i] = m->local_dims[i]; }}
    return success;
};

int spMeshGlobalOffset(spMesh const *m, size_type *dims, ptrdiff_t *offset)
{
    for (int i = 0; i < m->ndims; ++i)
    {
        dims[i] = m->global_dims[i];
        offset[i] = m->global_start[i] - m->local_start[i];
    }
    return SP_SUCCESS;
};
int spMeshGetStrides(spMesh const *m, size_type *res)
{
    if (res != NULL)
    {
        res[0] = 1;
        res[1] = m->local_dims[0];
        res[2] = m->local_dims[0] * m->local_dims[1];
    }
    return SP_SUCCESS;
}

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
                     int is_soa)
{


    int mesh_ndims = spMeshNDims(m);


    *array_ndims = spMeshNDims(m) + attr_ndims;

    if (is_soa == SP_TRUE)
    {
        for (int i = 0; i < attr_ndims; ++i)
        {
            l_dims[i] = attr_dims[i];
            l_start[i] = 0;
            l_count[i] = attr_dims[i];
        }
        *start_mesh_dim = attr_ndims;
    }
    else
    {
        for (int i = 0; i < attr_ndims; ++i)
        {
            l_dims[mesh_ndims + i] = attr_dims[i];
            l_start[mesh_ndims + i] = 0;
            l_count[mesh_ndims + i] = attr_dims[i];
        }
        *start_mesh_dim = 0;

    }


    SP_CHECK_RETURN(spMeshLocalDomain(m,
                                      domain_tag,
                                      l_dims + (*start_mesh_dim),
                                      l_start + (*start_mesh_dim),
                                      l_count + (*start_mesh_dim)));


    if (g_dims != NULL && g_start != NULL)
    {
        for (int i = 0; i < *array_ndims; ++i)
        {
            g_dims[i] = l_dims[i];
            g_start[i] = l_start[i];
        }

        ptrdiff_t offset[*array_ndims];

        SP_CHECK_RETURN(spMeshGlobalOffset(m, g_dims + (*start_mesh_dim), offset + (*start_mesh_dim)));

        for (int i = 0; i < mesh_ndims; ++i)
        {
            g_start[*start_mesh_dim + i] += offset[*start_mesh_dim + i];
        }
    };

    return SP_SUCCESS;

};

size_type spMeshHash(spMesh const *m, MeshEntityId id, int iform)
{
    return 0;
};


int spMeshWrite(const spMesh *ctx, const char *name, int flag)
{
    return SP_SUCCESS;
};

int spMeshRead(spMesh *ctx, const char *name, int flag)
{
    return SP_SUCCESS;

}