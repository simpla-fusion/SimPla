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
#include "spAlogorithm.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"
#include "spParticle.impl.h"


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

    spField *bucket_start, *bucket_end;
    size_type *sorted_id;
};

int spParticleCreate(spParticle **sp, const spMesh *mesh)
{
    int error_code = SP_SUCCESS;

    SP_CALL(spMeshAttributeCreate((spMeshAttribute **) sp,
                                  sizeof(spParticle), mesh, VOLUME));
    (*sp)->m_num_of_particle_ = 0;
    (*sp)->m_max_num_of_particle_ = 0;
    (*sp)->m_pic_ = 0;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->charge = 1;
    (*sp)->mass = 1;
    (*sp)->is_sorted = SP_FALSE;
    SP_CALL(spFieldCreate(&((*sp)->bucket_start), mesh, VOLUME, SP_TYPE_size_type));
    SP_CALL(spFieldCreate(&((*sp)->bucket_end), mesh, VOLUME, SP_TYPE_size_type));

    return error_code;

}

int spParticleDeploy(spParticle *sp)
{
    int error_code = SP_SUCCESS;

    if (sp == NULL) { return SP_DO_NOTHING; }

    size_type num_of_cell = spMeshGetNumberOfEntities(spMeshAttributeGetMesh((spMeshAttribute *) (sp)), SP_DOMAIN_ALL,
                                                      spMeshAttributeGetForm((spMeshAttribute *) (sp)));
    sp->m_max_num_of_particle_ = num_of_cell * sp->m_pic_ * 3 / 2;

    SP_CALL(spFieldClear(sp->bucket_start));
    SP_CALL(spFieldClear(sp->bucket_end));

    SP_CALL(spParallelDeviceAlloc((void **) &(sp->sorted_id),
                                  sp->m_max_num_of_particle_ * sizeof(uint)));

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        spParallelDeviceAlloc(&(sp->m_attrs_[i].data),
                              spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * sp->m_max_num_of_particle_);
    }
    void *d[spParticleGetNumberOfAttributes(sp)];
    SP_CALL(spParticleGetAllAttributeData(sp, d));
    SP_CALL(spParallelDeviceAlloc((void **) &(sp->m_current_data_),
                                  spParticleGetNumberOfAttributes(sp) * sizeof(void *)));
    SP_CALL(spParallelMemcpy(sp->m_current_data_, d,
                             spParticleGetNumberOfAttributes(sp) * sizeof(void *)));


    return error_code;
}

int spParticleDestroy(spParticle **sp)
{
    if (sp == NULL || *sp == NULL) { return SP_DO_NOTHING; }
    int error_code = SP_SUCCESS;
    SP_CALL(spFieldDestroy(&(*sp)->bucket_start));
    SP_CALL(spFieldDestroy(&(*sp)->bucket_end));

    SP_CALL(spParallelDeviceFree((void **) &((*sp)->sorted_id)));

    for (int i = 0; i < (*sp)->m_num_of_attrs_; ++i)
    {
        SP_CALL(spParallelDeviceFree(&((*sp)->m_attrs_[i].data)));
        SP_CALL(spDataTypeDestroy(&((*sp)->m_attrs_[i].data_type)));
    }

    SP_CALL(spParallelDeviceFree((void **) &((*sp)->m_current_data_)));

    SP_CALL(spMeshAttributeDestroy((spMeshAttribute **) sp));


    return error_code;
}

int spParticleInitialize(spParticle *sp, int const *dist_types)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute *) sp);

    size_type num_of_pic = spParticleGetPIC(sp);

    size_type max_number_of_particle = spParticleGetMaxNumOfParticle(sp);

    int num_of_dimensions = spParticleGetNumberOfAttributes(sp);

    int l_dist_types[num_of_dimensions];

    for (int i = 0; i < 6; ++i) { l_dist_types[i] = dist_types == NULL ? SP_RAND_UNIFORM : dist_types[i]; }

    void *data[spParticleGetNumberOfAttributes(sp)];

    SP_CALL(spParticleGetAllAttributeData(sp, data));

    SP_CALL(spParallelMemset(((particle_head *) data)->id, -1,
                             max_number_of_particle * sizeof(int)));

    size_type x_min[3], x_max[3], strides[3];

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, x_min, x_max, NULL));

    SP_CALL(spMeshGetStrides(m, strides));

    sp->m_num_of_particle_ = spMeshGetNumberOfEntities(m, SP_DOMAIN_CENTER, iform) * num_of_pic;

    spRandomGenerator *sp_gen;

    size_type offset = 0;
    size_type total = sp->m_num_of_particle_;

    SP_CALL(spMPIPrefixSum(&offset, &total));

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

    return error_code;

}

/** meta data @{*/

int spParticleSetPIC(spParticle *sp, unsigned int pic)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    sp->m_pic_ = pic;

    return SP_SUCCESS;
}

unsigned int spParticleGetPIC(spParticle const *sp) { return sp->m_pic_; }

unsigned int spParticleGetMaxPIC(spParticle const *sp) { return sp->m_pic_ * 2; }

size_type spParticleGetNumOfParticle(const spParticle *sp) { return sp->m_num_of_particle_; }

size_type spParticleGetMaxNumOfParticle(const spParticle *sp) { return sp->m_max_num_of_particle_; }

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

Real spParticleGetMass(spParticle const *sp) { if (sp != NULL) { return sp->mass; } else { return 0; }}

Real spParticleGetCharge(spParticle const *sp) { if (sp != NULL) { return sp->charge; } else { return 0; }}

size_type spParticleGetSize(spParticle const *sp) { return sp->m_num_of_particle_; };

size_type spParticleGetCapacity(spParticle const *sp) { return sp->m_max_num_of_particle_; }
/** @} */
/** attribute @{*/
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

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    int error_code =
            SP_CALL(spDataTypeCreate(&(sp->m_attrs_[sp->m_num_of_attrs_].data_type), tag, size));

    sp->m_attrs_[sp->m_num_of_attrs_].offset = offset;
    strcpy(sp->m_attrs_[sp->m_num_of_attrs_].name, name);
    sp->m_attrs_[sp->m_num_of_attrs_].data = NULL;

    ++(sp->m_num_of_attrs_);
    return error_code;

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
/**   @}*/


/**  ID  and sort @{*/

int spParticleSort(spParticle *sp)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    uint iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type numParticles = spParticleGetNumOfParticle(sp);

    size_type *hash = (size_type *) spParticleGetAttributeData(sp, 0);

    size_type *start_pos, *end_pos, *index;

    int error_code = SP_SUCCESS;

    SP_CALL(spParticleGetBucketIndex(sp, &start_pos, &end_pos, &index));

    SP_CALL(sort_by_key(hash, hash + numParticles, index));

    return error_code;
};

int spParticleBuildBucket(spParticle *sp)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    int error_code = SP_SUCCESS;

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    int ndims = spMeshGetNDims(m);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type num_of_particle = spParticleGetSize(sp);

    SP_CALL(spParticleBuildBucketFromIndex(sp));

    size_type *hash = (size_type *) spParticleGetAttributeData(sp, 0);

    spField *bucket_count;

    SP_CALL(spFieldCreate(&bucket_count, m, VOLUME, SP_TYPE_size_type));

    SP_CALL(spFieldDeploy(bucket_count));

    SP_CALL(spTransformMinus((size_type *) spFieldData(bucket_count),
                             spFieldData(sp->bucket_end),
                             spFieldData(sp->bucket_start),
                             num_of_cell));
    SP_CALL(spFieldSync(bucket_count));

    size_type *start_pos = NULL, *end_pos = NULL, *count = NULL, *sorted_id = NULL;
    size_type *index = sp->sorted_id;

    SP_CALL(spFieldCopyToHost((void **) &start_pos, sp->bucket_start));
    SP_CALL(spFieldCopyToHost((void **) &end_pos, sp->bucket_end));
    SP_CALL(spFieldCopyToHost((void **) &count, bucket_count));


    SP_CALL(spParallelHostAlloc((void **) &sorted_id, num_of_particle * sizeof(size_type)));
    SP_CALL(spParallelMemcpy((void *) sorted_id,
                             index, num_of_particle * sizeof(size_type)));

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_end[ndims + 1];
    size_type l_strides[ndims + 1];
    size_type l_count[ndims + 1];

    SP_CALL(spMeshGetLocalDims(m, l_dims));
    SP_CALL(spMeshGetStrides(m, l_strides));
    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, l_start, l_end, l_count));

    for (int i = 0; i < l_dims[0]; ++i)
        for (int j = 0; j < l_dims[1]; ++j)
            for (int k = 0; k < l_dims[2]; ++k)
            {
                if ((i >= l_start[0] && i < l_end[0]) &&
                    (j >= l_start[1] && j < l_end[1]) &&
                    (k >= l_start[2] && k < l_end[2])) { continue; }

                size_type s = i * l_strides[0] + j * l_strides[1] + k * l_strides[2];
                start_pos[s] = num_of_particle;
                num_of_particle += count[s];
                end_pos[s] = num_of_particle;
            }

    sp->m_num_of_particle_ = num_of_particle;

    SP_CALL(spFieldCopyToDevice(sp->bucket_start, start_pos));
    SP_CALL(spFieldCopyToDevice(sp->bucket_end, end_pos));

    SP_CALL(spParallelHostFree((void **) &start_pos));
    SP_CALL(spParallelHostFree((void **) &end_pos));
    SP_CALL(spParallelHostFree((void **) &count));
    SP_CALL(spParallelHostFree((void **) &sorted_id));

    SP_CALL(spParticleResetHash(sp));
    SP_CALL(spFieldDestroy(&bucket_count));

    return error_code;
}

int spParticleRearrange(spParticle *sp)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    int error_code = SP_SUCCESS;
    size_type numParticles = spParticleGetNumOfParticle(sp);

    size_type *start_pos, *end_pos, *index;

    SP_CALL(spParticleGetBucketIndex(sp, &start_pos, &end_pos, &index));

    Real *buffer = NULL;

    SP_CALL(spParallelDeviceAlloc((void **) &buffer, sizeof(Real) * spParticleGetCapacity(sp)));

    for (int i = 1; i < spParticleGetNumberOfAttributes(sp); ++i)
    {
        Real *dest = buffer;
        buffer = (Real *) spParticleGetAttributeData(sp, i);
        SP_CALL(spMemoryRelativeCopy(dest, buffer, numParticles, index));
        SP_CALL(spParticleSetAttributeData(sp, i, dest));
    }


    SP_CALL(spParallelDeviceFree((void **) &buffer));

    SP_CALL(spFillSeqInt(index, spParticleGetCapacity(sp), 0));

    return error_code;
}

int spParticleGetBucketIndex(spParticle *sp, size_type **start_pos, size_type **end_pos, size_type **index)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    if (start_pos != NULL) { *start_pos = spFieldData(sp->bucket_start); }
    if (end_pos != NULL) { *end_pos = spFieldData(sp->bucket_end); }
    if (index != NULL) { *index = sp->sorted_id; }
    return SP_SUCCESS;
}

/**
 *
 * @param sp
 */
int spParticleSync(spParticle *sp)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    int error_code = SP_SUCCESS;
    SP_CALL(spParticleSort(sp));

    SP_CALL(spParticleBuildBucket(sp));

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    int ndims = spMeshGetNDims(m);

    size_type num_of_cell = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform);

    size_type num_of_particle = spParticleGetSize(sp);

    size_type *start_pos = NULL, *end_pos = NULL, *sorted_id = NULL;

    SP_CALL(spFieldCopyToHost((void **) &start_pos, sp->bucket_start));

    SP_CALL(spFieldCopyToHost((void **) &end_pos, sp->bucket_end));

    SP_CALL(spParallelHostAlloc((void **) &sorted_id, num_of_particle * sizeof(size_type)));

    SP_CALL(spParallelMemcpy((void *) sorted_id,
                             sp->sorted_id, num_of_particle * sizeof(size_type)));

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_end[ndims + 1];
    size_type l_count[ndims + 1];

    SP_CALL(spMeshGetLocalDims(m, l_dims));
    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, l_start, l_end, l_count));

    /* MPI COMM Start */

    spDataType *d_type;

    SP_CALL(spDataTypeCreate(&d_type, SP_TYPE_Real, sizeof(Real)));

    spMPICartUpdater *updater;

    SP_CALL(spMPICartUpdaterCreate(&updater,
                                   spMPIComm(),
                                   d_type,
                                   0,
                                   ndims,
                                   l_dims,
                                   l_start,
                                   NULL,
                                   l_count,
                                   NULL,
                                   start_pos,
                                   end_pos,
                                   sorted_id));

    void *d[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    SP_CALL(spParticleGetAllAttributeData(sp, d));

    SP_CALL(spMPICartUpdateAll(updater, spParticleGetNumberOfAttributes(sp) - 1, d + 1));

    SP_CALL(spMPICartUpdaterDestroy(&updater));

    SP_CALL(spDataTypeDestroy(&d_type));

    /* MPI COMM End*/
    SP_CALL(spParallelHostFree((void **) &start_pos));
    SP_CALL(spParallelHostFree((void **) &end_pos));
    SP_CALL(spParallelHostFree((void **) &sorted_id));


    return error_code;

}

/**  @}*/
int
spParticleWrite(spParticle *sp, spIOStream *os, const char *name, int flag)
{
    if (sp == NULL) { return SP_DO_NOTHING; }

    int error_code = SP_SUCCESS;

    SP_CALL(spParticleRearrange(sp));

    SP_CALL(spParticleCoordinateLocalToGlobal(sp));

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

    size_type local_count = spParticleGetNumOfParticle(sp);
    size_type local_offset = 0;
    size_type global_offset, global_count = local_count;

    SP_CALL(spMPIPrefixSum(&global_offset, &global_count));


    size_type total_size_in_byte = 0;

    void *buffer = NULL;

    for (int i = 1; i < sp->m_num_of_attrs_; ++i)
    {
        size_type new_total_size_in_byte = spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * local_count;

        if (new_total_size_in_byte != total_size_in_byte)
        {
            SP_CALL(spParallelHostFree(&buffer));
            total_size_in_byte = new_total_size_in_byte;
            SP_CALL(spParallelHostAlloc(&buffer, total_size_in_byte));
        }

        SP_CALL(spMemoryDeviceToHost(&buffer, sp->m_attrs_[i].data, total_size_in_byte));

        SP_CALL(spIOStreamWriteSimple(os,
                                      sp->m_attrs_[i].name,
                                      sp->m_attrs_[i].data_type,
                                      buffer,
                                      1,
                                      &local_count,
                                      &local_offset,
                                      NULL,
                                      &local_count,
                                      NULL,
                                      &global_count,
                                      &global_offset,
                                      flag));
    }
    SP_CALL(spParallelHostFree(&buffer));
    SP_CALL(spIOStreamOpen(os, curr_path));
    return error_code;

}

int spParticleRead(struct spParticle_s *sp, spIOStream *os, const char *url, int flag)
{
    if (sp == NULL) { return SP_DO_NOTHING; }
    UNIMPLEMENTED;

    return SP_UNIMPLEMENTED;
}

