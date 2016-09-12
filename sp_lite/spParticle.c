//
// Created by salmon on 16-7-20.
//
#include "sp_lite_def.h"

#include <string.h>
#include <assert.h>

#include "spObject.h"
#include "spDataType.h"
#include "spParallel.h"
#include "spMPI.h"
#include "spIOStream.h"
#include "spRandom.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"
#include "../../../../../usr/local/cuda/include/driver_types.h"

#ifndef SP_MAX_NUMBER_OF_PARTICLE_ATTR
#    define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16
#endif


typedef struct spParticleAttrEntity_s
{
    spDataType *data_type;
    size_type offset;
    char name[255];
    void *data;
} spParticleAttrEntity;

/**
 *   Particle (phase space distribution function):
 *     - '''fiber bundle (P)''' on the '''base manifold (M)'''
 *
 *   fiber :
 *     - particles in a same cell(simplex) on '''M'''
 *     - link of '''page'''s
 *   page :
 *     - a group of particle ;
 *     - number of particles in group is a constant SP_NUMBER_OF_ENTITIES_IN_PAGE (128)
 *
 */
struct spParticle_s
{
    SP_MESH_ATTR_HEAD

    Real mass;

    Real charge;

    unsigned int m_pic_;

    size_type m_max_hash_;

    size_type m_num_of_particle_;

    size_type m_max_num_of_particle_;

    unsigned int m_num_of_attrs_;

    spParticleAttrEntity m_attrs_[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    void **m_current_data_;
    int is_sorted;

    spField *num_pic, *start_pos, *end_pos;
    uint *sorted_id;
};

int spParticleCreate(spParticle **sp, const spMesh *mesh)
{
    SP_CALL(spMeshAttributeCreate((spMeshAttribute **) sp, sizeof(spParticle), mesh, VOLUME));
    (*sp)->m_num_of_particle_ = 0;
    (*sp)->m_max_num_of_particle_ = 0;
    (*sp)->m_pic_ = 0;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->charge = 1;
    (*sp)->mass = 1;
    (*sp)->is_sorted = SP_FALSE;
    spFieldCreate(&(*sp)->start_pos, mesh, VOLUME, SP_TYPE_uint);
    spFieldCreate(&(*sp)->end_pos, mesh, VOLUME, SP_TYPE_uint);
    spFieldCreate(&(*sp)->num_pic, mesh, VOLUME, SP_TYPE_uint);

    return SP_SUCCESS;

}

int spParticleDeploy(spParticle *sp)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    size_type num_of_cell = spMeshGetNumberOfEntities(spMeshAttributeGetMesh((spMeshAttribute *) (sp)), SP_DOMAIN_ALL,
                                                      spMeshAttributeGetForm((spMeshAttribute *) (sp)));
    sp->m_max_num_of_particle_ = num_of_cell * sp->m_pic_ * 3 / 2;

    SP_CALL(spParallelDeviceAlloc((void **) &(sp->start_pos), num_of_cell * sizeof(uint)));
    SP_CALL(spParallelDeviceAlloc((void **) &(sp->end_pos), num_of_cell * sizeof(uint)));
    SP_CALL(spParallelDeviceAlloc((void **) &(sp->num_pic), num_of_cell * sizeof(uint)));
    SP_CALL(spParallelMemset((void *) (sp->start_pos), 0, num_of_cell * sizeof(uint)));
    SP_CALL(spParallelMemset((void *) (sp->end_pos), 0, num_of_cell * sizeof(uint)));
    SP_CALL(spParallelMemset((void *) (sp->num_pic), 0, num_of_cell * sizeof(uint)));

    SP_CALL(spParallelDeviceAlloc((void **) &(sp->sorted_id), sp->m_max_num_of_particle_ * sizeof(uint)));

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        spParallelDeviceAlloc(&(sp->m_attrs_[i].data),
                              spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * sp->m_max_num_of_particle_);
    }
    void *d[spParticleGetNumberOfAttributes(sp)];
    SP_CALL(spParticleGetAllAttributeData(sp, d));
    SP_CALL(spParallelDeviceAlloc((void **) &(sp->m_current_data_),
                                  spParticleGetNumberOfAttributes(sp) * sizeof(void *)));
    SP_CALL(spParallelMemcpy(sp->m_current_data_, d, spParticleGetNumberOfAttributes(sp) * sizeof(void *)));

    SP_CALL(spFieldClear(sp->start_pos));
    SP_CALL(spFieldClear(sp->end_pos));
    SP_CALL(spFieldClear(sp->num_pic));

    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (sp == NULL || *sp == NULL) { return SP_DO_NOTHING; }

    SP_CALL(spFieldDestroy(&(*sp)->start_pos));
    SP_CALL(spFieldDestroy(&(*sp)->end_pos));
    SP_CALL(spFieldDestroy(&(*sp)->num_pic));

    SP_CALL(spParallelDeviceFree((void **) &((*sp)->sorted_id)));

    for (int i = 0; i < (*sp)->m_num_of_attrs_; ++i)
    {
        spParallelDeviceFree(&((*sp)->m_attrs_[i].data));
        SP_CALL(spDataTypeDestroy(&((*sp)->m_attrs_[i].data_type)));
    }

    spParallelDeviceFree((void **) &((*sp)->m_current_data_));

    SP_CALL(spMeshAttributeDestroy((spMeshAttribute **) sp));


    return SP_SUCCESS;
}

int spParticleDeepSort(spParticle *sp)
{
    spParticleSort(sp);

    sp->is_sorted = SP_TRUE;

//    UNIMPLEMENTED;

    return SP_SUCCESS;
}

int spParticleIsSorted(spParticle const *sp) { return sp->is_sorted; }

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    SP_CALL(spDataTypeCreate(&(sp->m_attrs_[sp->m_num_of_attrs_].data_type), tag, size));

    sp->m_attrs_[sp->m_num_of_attrs_].offset = offset;
    strcpy(sp->m_attrs_[sp->m_num_of_attrs_].name, name);
    sp->m_attrs_[sp->m_num_of_attrs_].data = NULL;

    ++(sp->m_num_of_attrs_);
}

int spParticleGetNumberOfAttributes(spParticle const *sp) { return sp->m_num_of_attrs_; }

int spParticleGetAttributeName(spParticle *sp, int i, char *name)
{
    strcpy(name, sp->m_attrs_[i].name);
    return SP_SUCCESS;
};

size_type spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i)
{
    return spDataTypeSizeInByte(sp->m_attrs_[i].data_type);
};

void *spParticleGetAttributeData(spParticle *sp, int i) { return sp->m_attrs_[i].data; }

int spParticleSetAttributeData(spParticle *sp, int i, void *data)
{
    assert(i < sp->m_num_of_attrs_);
    sp->m_attrs_[i].data = data;
    return SP_SUCCESS;
}

int spParticleInitialize(spParticle *sp, int const *dist_types)
{
    if (sp == NULL) { return SP_DO_NOTHING; }


    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute *) sp);

    size_type num_of_pic = spParticleGetPIC(sp);

    size_type max_number_of_particle = spParticleGetMaxNumOfParticle(sp);

    int num_of_dimensions = spParticleGetNumberOfAttributes(sp);

    int l_dist_types[num_of_dimensions];

    for (int i = 0; i < 6; ++i) { l_dist_types[i] = dist_types == NULL ? SP_RAND_UNIFORM : dist_types[i]; }

    void *data[spParticleGetNumberOfAttributes(sp)];

    SP_CALL(spParticleGetAllAttributeData(sp, data));

    SP_CALL(spParallelMemset(((particle_head *) data)->id, -1, max_number_of_particle * sizeof(int)));

    size_type x_min[3], x_max[3], strides[3];

    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_CENTER, x_min, x_max, strides));

    sp->m_num_of_particle_ = spMeshGetNumberOfEntities(m, SP_DOMAIN_CENTER, iform) * num_of_pic;

    spRandomGenerator *sp_gen;

    size_type offset = sp->m_num_of_particle_;

    SP_CALL(spMPIPrefixSum(&offset, NULL));

    SP_CALL(spRandomGeneratorCreate(&sp_gen, SP_RAND_GEN_SOBOL, 6, offset));

    strides[0] *= num_of_pic;
    strides[1] *= num_of_pic;
    strides[2] *= num_of_pic;

    SP_CALL(spRandomMultiDistributionInCell(sp_gen,
                                            l_dist_types,
                                            (Real **) (data + 1),
                                            x_min,
                                            x_max,
                                            strides,
                                            num_of_pic));

    SP_CALL(spRandomGeneratorDestroy(&sp_gen));

}

int spParticleSetPIC(spParticle *sp, unsigned int pic)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    sp->m_pic_ = pic;

    return SP_SUCCESS;
}

unsigned int spParticleGetPIC(spParticle const *sp) { return sp->m_pic_; }

size_type spParticleGetNumOfParticle(const spParticle *sp) { return sp->m_num_of_particle_; }

size_type spParticleGetMaxNumOfParticle(const spParticle *sp) { return sp->m_max_num_of_particle_; }

int spParticleGetAllAttributeData(spParticle *sp, void **res)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    for (int i = 0, ie = spParticleGetNumberOfAttributes(sp); i < ie; ++i)
    {
        res[i] = spParticleGetAttributeData(sp, i);
    }
    return SP_SUCCESS;
};

int spParticleGetAllAttributeData_device(spParticle *sp, void ***data)
{


    if (sp == NULL)
    {
        *data = NULL;

        return SP_DO_NOTHING;
    } else
    {
        *data = sp->m_current_data_;

        return SP_SUCCESS;
    }
}

int spParticleSetMass(spParticle *sp, Real m)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    sp->mass = m;

    return SP_SUCCESS;
}

int spParticleSetCharge(spParticle *sp, Real e)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    sp->charge = e;

    return SP_SUCCESS;
}

Real spParticleGetMass(spParticle const *sp) { return sp->mass; }

Real spParticleGetCharge(spParticle const *sp) { return sp->charge; }

size_type spParticleGetSize(spParticle const *sp) { return sp->m_num_of_particle_; };

size_type spParticlePush(spParticle *sp, size_type s) { return sp->m_num_of_particle_ += s; };

size_type spParticleGetCapacity(spParticle const *sp) { return sp->m_max_num_of_particle_; }

int spParticleGetIndexArray(spParticle *sp, uint **start_pos, uint **end_pos, uint **index)
{
    *start_pos = spFieldData(sp->start_pos);
    *end_pos = spFieldData(sp->end_pos);
    *index = sp->sorted_id;
    return SP_SUCCESS;
}

/**
 *
 * @param sp
 */
int spParticleSync(spParticle *sp)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    sp->is_sorted = SP_FALSE;

    SP_CALL(spParticleSort(sp));

    SP_CALL(spFieldSync(sp->start_pos));
    SP_CALL(spFieldSync(sp->end_pos));
    SP_CALL(spFieldSync(sp->num_pic));

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    int ndims = spMeshGetNDims(m);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type num_of_particle = spParticleGetSize(sp);

    uint *start_pos = NULL, *end_pos = NULL, *num_pic = NULL, *sorted_id = NULL;

    SP_CALL(spFieldCopyToHost((void **) &start_pos, sp->start_pos));
    SP_CALL(spFieldCopyToHost((void **) &end_pos, sp->end_pos));
    SP_CALL(spFieldCopyToHost((void **) &num_pic, sp->num_pic));


    SP_CALL(spParallelHostAlloc((void **) &sorted_id, num_of_particle * sizeof(uint)));
    SP_CALL(spParallelMemcpy((void *) sorted_id, sp->sorted_id, num_of_particle * sizeof(uint)));

    size_type l_dims[ndims + 1];
    size_type start[ndims + 1];
    size_type l_end[ndims + 1];
    size_type l_strides[ndims + 1];

    spMeshGetDims(m, l_dims);
    spMeshGetArrayShape(m, SP_DOMAIN_CENTER, start, l_end, l_strides);
    for (int i = 0; i < l_dims[0]; ++i)
        for (int j = 0; j < l_dims[1]; ++j)
            for (int k = 0; k < l_dims[2]; ++k)
            {
                if ((i >= start[0] && i < l_end[0]) &&
                    (j >= start[1] && j < l_end[1]) &&
                    (k >= start[2] && k < l_end[2])) { continue; }

                size_type s = i * l_strides[0] + j * l_strides[1] + k * l_strides[2];
                start_pos[s] = (uint) sp->m_num_of_particle_;
                sp->m_num_of_particle_ += s;
                end_pos[s] = start_pos[s] + num_pic[s];
            }

    SP_CALL(spFieldCopyToDevice(sp->start_pos, (void **) &start_pos));
    SP_CALL(spFieldCopyToDevice(sp->end_pos, (void **) &end_pos));
    SP_CALL(spFieldCopyToDevice(sp->num_pic, (void **) &num_pic));




    /* MPI COMM Start */

    MPI_Comm comm = spMPIComm();

    if (comm == MPI_COMM_NULL) { return SP_DO_NOTHING; }

    int tope_type = MPI_CART;

    MPI_Topo_test(comm, &tope_type);

    assert(tope_type == MPI_CART);


    int mpi_topology_ndims = 0;
    int mpi_topo_dims[3];
    int periods[3];
    int mpi_topo_coord[3];
    spMPITopology(&mpi_topology_ndims,
                  mpi_topo_dims,
                  periods,
                  mpi_topo_coord);

    assert(mpi_topology_ndims <= ndims);

    int num_of_neighbour = 2 * mpi_topology_ndims;

    int dims[ndims];

//
//
////    SP_CALL(spMeshGetGlobalArrayShape(m, SP_DOMAIN_CENTER,
////                                      (iform == VERTEX || iform == VOLUME) ? 0 : 1,
////                                      &num_of_sub, &array_ndims, &mesh_start_dim, NULL, NULL,
////                                      l_dims, start, l_count, spFieldIsSoA(f)));
//
//    for (int i = 0; i < ndims; ++i) { dims[i] = (int) l_dims[i]; }
//
//    int s_count_lower[ndims];
//    int s_start_lower[ndims];
//    int s_count_upper[ndims];
//    int s_start_upper[ndims];
//
//    int r_count_lower[ndims];
//    int r_start_lower[ndims];
//    int r_count_upper[ndims];
//    int r_start_upper[ndims];
//
//    for (int d = 0; d < mpi_topology_ndims; ++d)
//    {
//
//
//        if (dims[d] == 1) { continue; }
//
//
//        for (int i = 0; i < ndims; ++i)
//        {
//            if (i < d)
//            {
//                s_count_lower[i] = dims[i];
//                s_start_lower[i] = 0;
//                s_count_upper[i] = dims[i];
//                s_start_upper[i] = 0;
//
//                r_count_lower[i] = dims[i];
//                r_start_lower[i] = 0;
//                r_count_upper[i] = dims[i];
//                r_start_upper[i] = 0;
//            } else if (i == d)
//            {
//                s_count_lower[i] = (int) start[i];
//                s_start_lower[i] = (int) start[i];
//                s_count_upper[i] = (int) (dims[i] - count[i] - start[i]);
//                s_start_upper[i] = (int) (start[i] + count[i] - s_count_upper[i]);
//
//                r_count_lower[i] = (int) start[i];
//                r_start_lower[i] = (int) 0;
//                r_count_upper[i] = (int) (dims[i] - count[i] - start[i]);
//                r_start_upper[i] = (int) dims[i] - s_count_upper[i];
//            } else
//            {
//                s_count_lower[i] = (int) count[i];
//                s_start_lower[i] = (int) start[i];
//                s_count_upper[i] = (int) count[i];
//                s_start_upper[i] = (int) start[i];
//
//                r_count_lower[i] = (int) count[i];
//                r_start_lower[i] = (int) start[i];
//                r_count_upper[i] = (int) count[i];
//                r_start_upper[i] = (int) start[i];
//            };
//        }
//
//
//    }

    /* MPI COMM End*/
    SP_CALL(spParallelHostFree((void **) &start_pos));
    SP_CALL(spParallelHostFree((void **) &sorted_id));
    SP_CALL(spParallelHostFree((void **) &num_pic));


    return SP_SUCCESS;

}

int
spParticleWrite(spParticle const *sp, spIOStream *os, const char *name, int flag)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    assert(spParticleIsSorted(sp) == SP_TRUE);

    char curr_path[2048];
    char new_path[2048];
    strcpy(new_path, name);
    new_path[strlen(name)] = '/';
    new_path[strlen(name) + 1] = '\0';

    SP_CALL(spIOStreamPWD(os, curr_path));
    SP_CALL(spIOStreamOpen(os, new_path));

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);
    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);
    int ndims = spMeshGetNDims(m);

    size_type local_number = spParticleGetSize(sp);
    size_type offset = (int) local_number, total_num;

    SP_CALL(spMPIPrefixSum(&offset, &total_num));

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);
    spField *start, *end;

    spFieldCreate(&start, m, iform, SP_TYPE_uint);

    spFieldCreate(&end, m, iform, SP_TYPE_uint);


    spFieldDeploy(start);

    spFieldDeploy(end);

    uint *start_pos, *end_pos, *index;

    spParticleGetIndexArray((spParticle *) sp, &start_pos, &end_pos, &index);

    spMemoryDeviceToHost(spFieldData(start), (void *) start_pos, num_of_cell);

    spMemoryDeviceToHost(spFieldData(end), (void *) end_pos, num_of_cell);

    spFieldAdd(start, &offset);

    spFieldAdd(end, &offset);

    spFieldWrite(start, os, "start_pos", SP_FILE_NEW);

    spFieldWrite(end, os, "end_pos", SP_FILE_NEW);

    spFieldDestroy(&start);

    spFieldDestroy(&end);

    size_type total_size_in_byte = 0;
    size_type local_offset = 0;
    void *buffer = NULL;

    for (int i = 1; i < sp->m_num_of_attrs_; ++i)
    {
        size_type new_total_size_in_byte = spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * local_number;

        if (new_total_size_in_byte != total_size_in_byte)
        {
            spMemoryHostFree(&buffer);
            total_size_in_byte = new_total_size_in_byte;
            spParallelHostAlloc(&buffer, total_size_in_byte);
        }

        spMemoryDeviceToHost(&buffer, sp->m_attrs_[i].data, total_size_in_byte);

        SP_CALL(spIOStreamWriteSimple(os,
                                      sp->m_attrs_[i].name,
                                      sp->m_attrs_[i].data_type,
                                      buffer,
                                      1,
                                      &local_number,
                                      &local_offset,
                                      NULL,
                                      &local_number,
                                      NULL,
                                      &total_num,
                                      &offset,
                                      flag));
    }
    spMemoryHostFree(&buffer);
    spFieldDestroy(&start);
    spFieldDestroy(&end);

    SP_CALL(spIOStreamOpen(os, curr_path));
    return SP_SUCCESS;

}

int spParticleRead(struct spParticle_s *sp, spIOStream *os, const char *url, int flag)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    UNIMPLEMENTED;

    return SP_UNIMPLEMENTED;
}

