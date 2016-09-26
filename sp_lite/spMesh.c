/*
 * spMesh.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "sp_lite_def.h"
#include "spMesh.h"
#include "spParallel.h"
#include "spMPI.h"

MeshEntityId spMeshEntityIdFromArray(size_type const *s)
{
    MeshEntityId id;
    id.x = (int16_t) (s[0]);
    id.y = (int16_t) (s[1]);
    id.z = (int16_t) (s[2]);
    return id;
}

MeshEntityId spMeshEntityIdShift(MeshEntityId id, ptrdiff_t const *s)
{
    id.x += (int16_t) (s[0] * 2);
    id.y += (int16_t) (s[1] * 2);
    id.z += (int16_t) (s[2] * 2);
    return id;
}

int spMeshAttributeCreate(spMeshAttribute **f, size_type size, spMesh const *mesh, uint iform)
{

    SP_CALL(spObjectCreate((spObject **) (f), size));
    (*f)->m = mesh;
    (*f)->iform = iform;
    return SP_SUCCESS;
};

int spMeshAttributeDestroy(spMeshAttribute **f)
{
    SP_CALL(spObjectDestroy((spObject **) (f)));
    return SP_SUCCESS;
};

spMesh const *spMeshAttributeGetMesh(spMeshAttribute const *f) { return f->m; }

uint spMeshAttributeGetForm(spMeshAttribute const *f) { return f->iform; };

struct spMesh_s
{

    int m_ndims_;
    size_type m_ghost_width_[4];
    size_type m_dims_[4];
    size_type m_start_[4];
    size_type m_count_[4];

    size_type m_global_dims_[4];
    size_type m_global_start_[4];

    Real m_global_coord_lower_[4];
    Real m_global_coord_upper[4];
    Real m_coord_lower[4];
    Real m_coord_upper[4];
    Real dx[4];
    Real inv_dx[4];

    size_type strides[3];

    int array_is_order_c;
};

int spMeshCreate(spMesh **m)
{
    *m = (spMesh *) malloc(sizeof(spMesh));
    (*m)->m_ndims_ = 3;

    (*m)->m_global_dims_[0] = 1;
    (*m)->m_global_dims_[1] = 1;
    (*m)->m_global_dims_[2] = 1;
    (*m)->m_global_dims_[3] = 3;

    (*m)->m_dims_[0] = 1;
    (*m)->m_dims_[1] = 1;
    (*m)->m_dims_[2] = 1;
    (*m)->m_dims_[3] = 3;

    (*m)->m_ghost_width_[0] = 0;
    (*m)->m_ghost_width_[1] = 0;
    (*m)->m_ghost_width_[2] = 0;

    (*m)->m_start_[0] = 0;
    (*m)->m_start_[1] = 0;
    (*m)->m_start_[2] = 0;
    (*m)->m_start_[3] = 0;


    (*m)->m_count_[0] = 1;
    (*m)->m_count_[1] = 1;
    (*m)->m_count_[2] = 1;
    (*m)->m_count_[3] = 3;


    (*m)->m_global_coord_lower_[0] = 0;
    (*m)->m_global_coord_lower_[1] = 0;
    (*m)->m_global_coord_lower_[2] = 0;

    (*m)->m_global_coord_upper[0] = 1;
    (*m)->m_global_coord_upper[1] = 1;
    (*m)->m_global_coord_upper[2] = 1;


    (*m)->m_coord_lower[0] = 0;
    (*m)->m_coord_lower[1] = 0;
    (*m)->m_coord_lower[2] = 0;

    (*m)->m_coord_upper[0] = 1;
    (*m)->m_coord_upper[1] = 1;
    (*m)->m_coord_upper[2] = 1;
    (*m)->array_is_order_c = SP_TRUE;
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

    self->m_ndims_ = 3;

    for (int i = 0; i < 3; ++i)
    {

        if (self->m_global_dims_[i] <= 1)
        {
            self->m_ghost_width_[i] = 0;

            self->m_global_dims_[i] = 1;
        }

        if (i < mpi_topo_ndims)
        {
            self->m_global_start_[i] = self->m_global_dims_[i] * mpi_topo_coords[i] / mpi_topo_dims[i];

            self->m_count_[i] =
                    self->m_global_dims_[i] * (mpi_topo_coords[i] + 1) / mpi_topo_dims[i] - self->m_global_start_[i];
        } else
        {
            self->m_global_start_[i] = 0;

            self->m_count_[i] = self->m_global_dims_[i];
        }

        self->m_start_[i] = self->m_ghost_width_[i];

        self->m_dims_[i] = self->m_count_[i] + self->m_ghost_width_[i] * 2;

        self->dx[i] = (self->m_global_coord_upper[i] - self->m_global_coord_lower_[i]) / self->m_global_dims_[i];

        self->m_coord_lower[i] = self->m_global_coord_lower_[i] + self->m_global_start_[i] * self->dx[i];

        self->m_coord_upper[i] = self->m_coord_lower[i] + self->m_count_[i] * self->dx[i];

        self->inv_dx[i] = (Real) ((self->m_global_dims_[i] <= 1) ? 0 : 1.0 / self->dx[i]);

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
//	spMemCopyToCache(SP_NEIGHBOUR_OFFSET, neighbour_offset, sizeof(neighbour_offset));
//	spMemCopyToCache(SP_NEIGHBOUR_OFFSET_flag, neighbour_flag, sizeof(neighbour_flag));
    SP_CALL(spMeshSetupParam(self));
    return SP_SUCCESS;

}

size_type spMeshGetNumberOfEntities(spMesh const *self, int tag, int iform)
{
    size_type res = (iform == VERTEX || iform == VOLUME) ? 1 : 3;
    switch (tag)
    {
        case SP_DOMAIN_CENTER:
            res *= self->m_count_[0] * self->m_count_[1] * self->m_count_[2];
            break;
        case SP_DOMAIN_ALL:
        default:
            res *= self->m_dims_[0] * self->m_dims_[1] * self->m_dims_[2];
            break;
    }

    return res;
}

void spMeshPoint(spMesh const *m, MeshEntityId id, Real *res)
{
    res[0] = m->m_coord_lower[0] + m->dx[0] * (id.x - (m->m_start_[0] << 1)) * 0.5;
    res[1] = m->m_coord_lower[1] + m->dx[1] * (id.y - (m->m_start_[1] << 1)) * 0.5;
    res[2] = m->m_coord_lower[2] + m->dx[2] * (id.z - (m->m_start_[2] << 1)) * 0.5;
};

int spMeshGetNDims(spMesh const *m) { return m->m_ndims_; };


Real spMeshCFLDtv(spMesh const *m, Real const *speed)
{
    Real const *dx = m->dx;

    Real res = 0;

    for (int i = 0; i < 3; ++i)
    {
        res += m->m_global_dims_[i] <= 1 ? 0 : (speed[i] / dx[i]) * (speed[i] / dx[i]);

    }
    return (Real) (0.5 / sqrt((double) (res)));
};

Real spMeshCFLDt(spMesh const *m, Real speed)
{
    Real v[3] = {speed, speed, speed};
    return spMeshCFLDtv(m, v);
}

#define GET_VEC3(_RES_, _ATTR_)                                   \
if (_RES_ == NULL) { return SP_FAILED; }                           \
else { for (int i = 0; i < 3; ++i) { _RES_[i] = m->_ATTR_[i]; }}       \
return SP_SUCCESS;                                               \

#define SET_VEC3(_RES_, _ATTR_)                                   \
if (_RES_ == NULL) { return SP_FAILED; }                             \
for (int i = 0; i < 3; ++i) { m->_ATTR_[i] = _RES_[i]; }     \
return SP_SUCCESS;                                                \


int spMeshSetDims(spMesh *m, size_type const *v) { SET_VEC3(v, m_global_dims_) }

int spMeshGetGlobalDims(spMesh const *m, size_type *v) { GET_VEC3(v, m_global_dims_) }

int spMeshGetDims(spMesh const *m, size_type *v) { GET_VEC3(v, m_dims_) }

int spMeshSetGhostWidth(spMesh *m, size_type const *v) { SET_VEC3(v, m_ghost_width_) }

int spMeshGetGhostWidth(spMesh const *m, size_type *res) { GET_VEC3(res, m_ghost_width_) }

int spMeshGetOrigin(spMesh const *m, Real *res) { GET_VEC3(res, m_coord_lower) }

int spMeshGetGlobalOrigin(spMesh const *m, Real *res) { GET_VEC3(res, m_global_coord_lower_) }

int spMeshGetDx(spMesh const *m, Real *res) { GET_VEC3(res, dx) }

int spMeshGetInvDx(spMesh const *m, Real *res) { GET_VEC3(res, inv_dx) }

#undef GET_VEC3
#undef SET_VEC3

int spMeshSetBox(spMesh *m, Real const *lower, Real const *upper)
{
    for (int i = 0; i < 3; ++i)
    {
        m->m_global_coord_lower_[i] = lower[i];
        m->m_global_coord_upper[i] = upper[i];
    }
    return SP_SUCCESS;

};

int spMeshGetBox(spMesh const *m, int tag, Real *lower, Real *upper)
{
    size_type start[3], count[3];

    spMeshGetDomain(m, tag, start, NULL, count);

    lower[0] = m->m_coord_lower[0] + ((int) start[0] - (int) (m->m_start_[0])) * m->dx[0];
    lower[1] = m->m_coord_lower[1] + ((int) start[1] - (int) (m->m_start_[1])) * m->dx[1];
    lower[2] = m->m_coord_lower[2] + ((int) start[2] - (int) (m->m_start_[2])) * m->dx[2];

    if (upper != NULL)
    {
        upper[0] = lower[0] + count[0] * m->dx[0];
        upper[1] = lower[1] + count[1] * m->dx[1];
        upper[2] = lower[2] + count[2] * m->dx[2];
    }

    return SP_SUCCESS;
};


int spMeshGetDomain(spMesh const *m, int tag, size_type *p_start, size_type *p_end, size_type *p_count)
{

    int success = SP_SUCCESS;
    int offset[3] = {0, 0, 0};
    size_type start[3] = {0, 0, 0};
    size_type end[3] = {0, 0, 0};
    size_type count[3] = {0, 0, 0};

    switch (tag)
    {
        case SP_DOMAIN_ALL:
            for (int i = 0; i < 3; ++i)
            {
                start[i] = 0;
                count[i] = m->m_dims_[i];
            }
            break;
        case SP_DOMAIN_CENTER:
            for (int i = 0; i < 3; ++i)
            {
                start[i] = m->m_start_[i];
                count[i] = m->m_count_[i];
            }

            break;
        case SP_DOMAIN_AFFECT_1:
            for (int i = 0; i < 3; ++i)
            {
                start[i] = m->m_dims_[i] <= 1 ? 0 : 1;
                count[i] = m->m_dims_[i] <= 1 ? 1 : m->m_dims_[i] - 2;
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
                        count[i] = m->m_ghost_width_[i];
                        if (m->m_ghost_width_[i] == 0) { success = SP_FAILED; }
                        break;
                    case 1:
                        start[i] = m->m_count_[i] + m->m_start_[i];
                        count[i] = m->m_dims_[i] - start[i];
                        if (m->m_ghost_width_[i] == 0) { success = SP_FAILED; }
                        break;
                    default: //0
                        start[i] = m->m_start_[i];
                        count[i] = m->m_count_[i];
                        break;
                }
            }
    }
    for (int i = 0; i < 3; ++i)
    {
        if (p_start != NULL) { p_start[i] = start[i]; }
        if (p_count != NULL) { p_count[i] = count[i]; }
        if (p_end != NULL) { p_end[i] = start[i] + count[i]; }

    }


    return success;
};


int spMeshGetStrides(spMesh const *m, size_type *res)
{
    if (res != NULL)
    {
        if (m->array_is_order_c == SP_TRUE)
        {
            res[2] = (m->m_global_dims_[2] == 1) ? 0 : 1;
            res[1] = (m->m_global_dims_[1] == 1) ? 0 : m->m_dims_[2];
            res[0] = (m->m_global_dims_[0] == 1) ? 0 : m->m_dims_[2] * m->m_dims_[1];
        } else
        {
            res[0] = (m->m_global_dims_[0] == 1) ? 0 : 1;
            res[1] = (m->m_global_dims_[1] == 1) ? 0 : m->m_dims_[0];
            res[2] = (m->m_global_dims_[2] == 1) ? 0 : m->m_dims_[0] * m->m_dims_[1];
        }
    }
    return SP_SUCCESS;
}

int spMeshGetGlobalStart(spMesh const *m, size_type *start)
{
    for (int i = 0; i < m->m_ndims_; ++i)
    {

        start[i] = m->m_global_start_[i];
    }
    return SP_SUCCESS;
};

int spMeshGetGlobalOffset(spMesh const *m, size_type *dims, ptrdiff_t *offset)
{
    for (int i = 0; i < m->m_ndims_; ++i)
    {
        dims[i] = m->m_global_dims_[i];
        offset[i] = m->m_global_start_[i] - m->m_start_[i];
    }
    return SP_SUCCESS;
};

int spMeshGetGlobalArrayShape(spMesh const *m, int domain_tag,
                              int attr_ndims, const size_type *attr_dims,
                              int *array_ndims,
                              int *start_mesh_dim,
                              size_type *g_dims,
                              size_type *g_start,
                              size_type *l_dims,
                              size_type *l_start,
                              size_type *l_count,
                              int is_soa)
{

    int mesh_ndims = spMeshGetNDims(m);

    *array_ndims = spMeshGetNDims(m) + attr_ndims;


    if (is_soa == SP_TRUE)
    {

        for (int i = 0; i < attr_ndims; ++i)
        {
            l_dims[i] = attr_dims[i];
            l_start[i] = 0;
            l_count[i] = attr_dims[i];
        }
        *start_mesh_dim = attr_ndims;
    } else
    {

        for (int i = 0; i < attr_ndims; ++i)
        {
            l_dims[mesh_ndims + i] = attr_dims[i];
            l_start[mesh_ndims + i] = 0;
            l_count[mesh_ndims + i] = attr_dims[i];
        }
        *start_mesh_dim = 0;
    }

    SP_CALL(spMeshGetDims(m, l_dims + (*start_mesh_dim)));
    SP_CALL(spMeshGetDomain(m, domain_tag, l_start + (*start_mesh_dim), NULL, l_count + (*start_mesh_dim)));


    if (g_dims != NULL && g_start != NULL)
    {
        for (int i = 0; i < *array_ndims; ++i)
        {
            g_dims[i] = l_dims[i];
            g_start[i] = l_start[i];
        }

        ptrdiff_t offset[*array_ndims];

        SP_CALL(spMeshGetGlobalOffset(m, g_dims + (*start_mesh_dim), offset + (*start_mesh_dim)));

        for (int i = 0; i < mesh_ndims; ++i) { g_start[*start_mesh_dim + i] += offset[*start_mesh_dim + i]; }
    };

    return SP_SUCCESS;

};

size_type spMeshHash(spMesh const *m, MeshEntityId id, int iform) { return 0; };

int spMeshWrite(const spMesh *ctx, const char *name) { return SP_FAILED; };

int spMeshRead(spMesh *ctx, const char *name) { return SP_FAILED; }

