//
// Created by salmon on 16-7-20.
//
#include "sp_lite_config.h"

#include <string.h>
#include <assert.h>

#include "spObject.h"
#include "spDataType.h"
#include "spParallel.h"
#include "spMPI.h"
#include "spIOStream.h"
#include "spAlogorithm.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"
#include "detail/spParticle.impl.h"
#include "detail/sp_device.h"

typedef struct spParticleAttrEntity_s
{
    int data_type;
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

    int m_data_type_tag_;

    unsigned int m_pic_;

    size_type m_max_hash_;

    size_type m_sorted_idx_tail_;

    size_type m_num_of_particle_;

    size_type m_max_num_of_particle_;

    unsigned int m_num_of_attrs_;

    spParticleAttrEntity m_attrs_[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    void **m_current_data_, **m_next_data_;
    int is_deployed;

    spField *bucket_start, *bucket_count;
    size_type *sorted_idx;
    size_type *cell_hash;

    size_type step_count;
    size_type defragment_freq;

    int need_sorting;

};

/** meta data @{*/


int spParticleEnableSorting(spParticle *sp)
{
    sp->need_sorting = SP_TRUE;
    return SP_SUCCESS;
}

int spParticleNeedSorting(spParticle const *sp)
{
    return sp->need_sorting;
};

int spParticleSetPIC(spParticle *sp, unsigned int pic)
{
    assert (sp != NULL);
    sp->m_pic_ = pic;
    return SP_SUCCESS;
}

unsigned int spParticleGetPIC(spParticle const *sp) { return sp->m_pic_; }

unsigned int spParticleGetMaxPIC(spParticle const *sp) { return sp->m_pic_ * 2; }

int spParticleSetMass(spParticle *sp, Real m)
{
    if (sp == NULL) { return SP_FAILED; }
    sp->mass = m;
    return SP_SUCCESS;
}

Real spParticleGetMass(spParticle const *sp) { if (sp != NULL) { return sp->mass; } else { return 1; }}

int spParticleSetCharge(spParticle *sp, Real e)
{
    if (sp == NULL) { return SP_FAILED; }
    sp->charge = e;
    return SP_SUCCESS;
}

Real spParticleGetCharge(spParticle const *sp) { if (sp != NULL) { return sp->charge; } else { return 1; }}

size_type spParticleSize(spParticle const *sp) { return sp->m_sorted_idx_tail_; };

int spParticleResize(spParticle *sp, size_type s)
{
    if (sp == NULL || s >= sp->m_max_num_of_particle_) { return SP_FAILED; }

    sp->m_sorted_idx_tail_ = s;

    return SP_SUCCESS;
};

size_type spParticleCapacity(spParticle const *sp) { return sp->m_max_num_of_particle_; }

int spParticleSetDefragmentFreq(spParticle *sp, size_t n)
{
    if (sp == NULL) { return SP_FAILED; }
    sp->defragment_freq = n;
    return SP_SUCCESS;
}


/** @} */

/** attribute @{*/
int spParticleGetAllAttributeData(spParticle *sp, void **res)
{
    if (sp == NULL) { return SP_FAILED; }

    for (int i = 0, ie = spParticleGetNumberOfAttributes(sp); i < ie; ++i)
    {
        res[i] = spParticleGetAttributeData(sp, i);
    }
    return SP_SUCCESS;
};

int spParticleGetAllAttributeData_device(spParticle *sp, void ***current_data, void ***next_data)
{
    if (sp == NULL) { return SP_FAILED; }

    if (current_data != NULL)
    {
        *current_data = sp->m_current_data_;
    }
    if (next_data != NULL) { *next_data = sp->m_next_data_; }

    return SP_SUCCESS;

}

int spParticleAddAttribute(spParticle *sp, const char name[], int type_tag)
{
    if (sp == NULL) { return SP_FAILED; }
    assert (sp->is_deployed == SP_FALSE);

    sp->m_attrs_[sp->m_num_of_attrs_].data_type = type_tag;
    strcpy(sp->m_attrs_[sp->m_num_of_attrs_].name, name);
    sp->m_attrs_[sp->m_num_of_attrs_].data = NULL;

    ++(sp->m_num_of_attrs_);
    return SP_SUCCESS;

}

int spParticleGetNumberOfAttributes(spParticle const *sp) { return sp->m_num_of_attrs_; }

int spParticleGetAttributeName(spParticle *sp, int i, char *name)
{
    assert (i < sp->m_num_of_attrs_);
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
#ifndef SP_DEFAULT_DEFRAGMENT_FREQ
#   define SP_DEFAULT_DEFRAGMENT_FREQ 10
#endif

int spParticleCreate(spParticle **sp, const spMesh *mesh)
{
    SP_CALL(spMeshAttributeCreate((spMeshAttribute **) sp, sizeof(spParticle), mesh, VOLUME));
    (*sp)->m_data_type_tag_ = SP_TYPE_Real;
    (*sp)->m_sorted_idx_tail_ = 0;
    (*sp)->m_max_num_of_particle_ = 0;
    (*sp)->m_pic_ = 0;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->charge = 1;
    (*sp)->mass = 1;
    (*sp)->is_deployed = SP_FALSE;
    (*sp)->step_count = 0;
    (*sp)->defragment_freq = SP_DEFAULT_DEFRAGMENT_FREQ;
    (*sp)->need_sorting = SP_FALSE;
    return SP_SUCCESS;

}

int spParticleDeploy(spParticle *sp)
{
    if (sp == NULL) { return SP_FAILED; }

    /* if  Particle is deployed, then  do nothing and return success*/
    if (sp->m_current_data_ != NULL) { return SP_SUCCESS; }

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    int iform = spMeshAttributeGetForm((spMeshAttribute *) sp);

    sp->m_sorted_idx_tail_ = spMeshGetNumberOfEntities(m, SP_DOMAIN_CENTER, iform) * sp->m_pic_;

    sp->m_max_num_of_particle_ = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform) * sp->m_pic_ * 2;

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        SP_CALL(spMemoryDeviceAlloc(&(sp->m_attrs_[i].data),
                                    spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * sp->m_max_num_of_particle_));
        SP_CALL(spMemorySet((sp->m_attrs_[i].data), 0,
                            spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * sp->m_max_num_of_particle_));
    }
    int num_of_attr = spParticleGetNumberOfAttributes(sp);
    void *d[num_of_attr];
    SP_CALL(spParticleGetAllAttributeData(sp, d));
    SP_CALL(spMemoryDeviceAlloc((void **) &(sp->m_current_data_), num_of_attr * sizeof(void *)));
    SP_CALL(spMemoryCopy(sp->m_current_data_, d, num_of_attr * sizeof(void *)));

    /* Deploy buckets */

    SP_CALL(spFieldCreate(&(sp->bucket_start), m, VOLUME, SP_TYPE_size_type));
    SP_CALL(spFieldCreate(&(sp->bucket_count), m, VOLUME, SP_TYPE_size_type));
    SP_CALL(spFieldClear(sp->bucket_start));
    SP_CALL(spFieldClear(sp->bucket_count));

    SP_CALL(spMemoryDeviceAlloc((void **) &(sp->sorted_idx), sp->m_max_num_of_particle_ * sizeof(size_type)));
    SP_CALL(spMemoryDeviceAlloc((void **) &(sp->cell_hash), sp->m_max_num_of_particle_ * sizeof(size_type)));


    sp->is_deployed = SP_TRUE;
    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (sp == NULL || *sp == NULL) { return SP_SUCCESS; }


    SP_CALL(spFieldDestroy(&(*sp)->bucket_start));
    SP_CALL(spFieldDestroy(&(*sp)->bucket_count));

    SP_CALL(spMemoryDeviceFree((void **) &((*sp)->sorted_idx)));
    SP_CALL(spMemoryDeviceFree((void **) &((*sp)->cell_hash)));

    for (int i = 0; i < (*sp)->m_num_of_attrs_; ++i) {SP_CALL(spMemoryDeviceFree(&((*sp)->m_attrs_[i].data))); }

    SP_CALL(spMemoryDeviceFree((void **) &((*sp)->m_current_data_)));

    SP_CALL(spMeshAttributeDestroy((spMeshAttribute **) sp));


    return SP_SUCCESS;
}

int spParticleInitialize(spParticle *sp, int const *dist_types)
{
    if (sp == NULL) { return SP_FAILED; }

    SP_CALL(spParticleDeploy(sp));

    /* Initialize particles*/

    void *data[spParticleGetNumberOfAttributes(sp)];

    SP_CALL(spParticleGetAllAttributeData(sp, data));

    size_type offset = 0, total = spParticleSize(sp);

    SP_CALL(spMPIPrefixSum(&offset, &total));

    SP_CALL(spParticleInitialize_device((Real **) (data), 6, dist_types, spParticleSize(sp), offset));

    SP_CALL(spParticleBucketInitialize_device(sp));

    return SP_SUCCESS;

}


size_type spParticleGlobalSize(spParticle const *sp)
{
    size_type total = sp->m_sorted_idx_tail_;

    SP_CALL(spMPIPrefixSum(NULL, &total));
    return total;
}

/**  ID  and sort @{*/

int spParticleGetBucket(spParticle *sp, size_type **start_pos, size_type **count, size_type **sorted_idx,
                        size_type **cell_hash)
{
    if (sp == NULL) { return SP_FAILED; }

    if (start_pos != NULL) { *start_pos = spFieldData(sp->bucket_start); }

    if (count != NULL) { *count = spFieldData(sp->bucket_count); }

    if (sorted_idx != NULL) { *sorted_idx = sp->sorted_idx; }

    if (cell_hash != NULL) { *cell_hash = sp->cell_hash; }

    return SP_SUCCESS;
}
//
//int spParticleGetBucket2(spParticle *sp, spField **start_pos, spField **count,
//                         size_type **sorted_idx, size_type **cell_hash)
//{
//    if (sp == NULL) { return SP_FAILED; }
//
//    if (start_pos != NULL) { *start_pos = sp->bucket_start; }
//
//    if (count != NULL) { *count = sp->bucket_count; }
//
//    if (sorted_idx != NULL) { *sorted_idx = sp->sorted_idx; }
//
//    if (cell_hash != NULL) { *cell_hash = sp->cell_hash; }
//
//    return SP_SUCCESS;
//}

int spParticleDefragment(spParticle *sp)
{
    if (sp == NULL) { return SP_FAILED; }

    size_type numParticles = spParticleSize(sp);

    size_type maxNumParticles = spParticleCapacity(sp);

    size_type *sorted_idx;

    SP_CALL(spParticleGetBucket(sp, NULL, NULL, &sorted_idx, NULL));

    Real *buffer = NULL;

    SP_CALL(spMemoryDeviceAlloc((void **) &buffer, sizeof(Real) * maxNumParticles));
    int num_of_attr = spParticleGetNumberOfAttributes(sp);

    for (int i = 0; i < num_of_attr; ++i)
    {
        Real *dest = buffer;

        buffer = (Real *) spParticleGetAttributeData(sp, i);

        SP_CALL(spMemoryCopyIndirect(dest, buffer, numParticles, sorted_idx));

        SP_CALL(spParticleSetAttributeData(sp, i, dest));
    }

    SP_CALL(spMemoryDeviceFree((void **) &buffer));
    void *d[num_of_attr];
    SP_CALL(spParticleGetAllAttributeData(sp, d));
    SP_CALL(spMemoryCopy(sp->m_current_data_, d, sizeof(void *) * num_of_attr));
    SP_CALL(spFillSeq(sorted_idx, SP_TYPE_size_type, maxNumParticles, 0, 1));

    return SP_SUCCESS;
}

int spParticleNextStep(spParticle *sp)
{
    void **t = sp->m_next_data_;

    sp->m_next_data_ = sp->m_current_data_;

    sp->m_current_data_ = t;

    if (sp->m_current_data_ != NULL) { return SP_SUCCESS; } else { return SP_FAILED; }
}

int spParticleSort(spParticle *sp)
{
    if (sp == NULL) { return SP_FAILED; }

    size_type numParticles = spParticleSize(sp);

    SP_CALL(sort_by_key(sp->cell_hash, sp->cell_hash + numParticles, sp->sorted_idx));

    ++sp->step_count;

    if (sp->step_count % sp->defragment_freq == 0) {SP_CALL(spParticleDefragment(sp)); }

    sp->need_sorting = SP_FALSE;

    SP_CALL(spParticleBucketBuild_device(sp));

    return SP_SUCCESS;
};


/**
 *
 * @param sp
 */
int spParticleSync(spParticle *sp)
{
    if (sp == NULL) { return SP_FAILED; }

    assert(spParticleNeedSorting(sp) == SP_FALSE);

    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute const *) sp);

    size_type *bucket_start, *bucket_count, *sorted_id;

    SP_CALL(spParticleGetBucket(sp, &bucket_start, &bucket_count, &sorted_id, NULL));

    void *d[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    int num_of_attr = spParticleGetNumberOfAttributes(sp);

    size_type num_of_particle = spParticleSize(sp);

    SP_CALL(spParticleGetAllAttributeData(sp, d));

    spMPIUpdater *updater = NULL;

    SP_CALL(spMeshGetMPIUpdater(m, &updater));

    SP_CALL(spMPIUpdateBucket(updater, sp->m_data_type_tag_, num_of_attr, d,
                              bucket_start, bucket_count, sorted_id, &num_of_particle));

    SP_CALL(spParticleResize(sp, num_of_particle));

    return SP_SUCCESS;
}
//    {
//        Real *buffer;
//        size_type strides[3];
//        SP_CALL(spMeshGetStrides(m, strides));
//        size_type num = spParticleSize(sp);
//        SP_CALL(spMemoryHostAlloc((void **) &buffer, num * sizeof(Real)));
//        SP_CALL(spMemoryCopy(buffer, spParticleGetAttributeData(sp, 5), num * sizeof(Real)));
//
//        printf("\n***************************************\n");
//
//        for (int i = 0; i < l_dims[0]; ++i)
//        {
//            printf("\n %4d|\t", i);
//            for (int j = 0; j < l_dims[1]; ++j)
//            {
//                for (int k = 0; k < l_dims[2]; ++k)
//                {
//                    size_type s = i * strides[0] + j * strides[1] + k * strides[2];
//
//                    printf(" %4.0f ", (buffer)[sorted_idx[bucket_start_pos[s] + 1]]);
//                }
//            }
//
//        }
//        printf("\n");
//        SP_CALL(spMemoryHostFree((void **) &buffer));
//    }

/**  @}*/
int
spParticleWrite(const spParticle *sp, spIOStream *os, const char *name, int flag)
{
    if (sp == NULL) { return SP_FAILED; }


//    SP_CALL(spParticleCoordinateLocalToGlobal(sp));

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

    size_type local_count = spParticleSize(sp);
    size_type local_offset = 0;
    size_type global_offset = 0, global_count = local_count;

    SP_CALL(spMPIPrefixSum(&global_offset, &global_count));

    size_type total_size_in_byte = 0;

    void *buffer = NULL;

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        size_type new_total_size_in_byte = spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * local_count;

        if (new_total_size_in_byte != total_size_in_byte)
        {
            SP_CALL(spMemoryHostFree(&buffer));

            total_size_in_byte = new_total_size_in_byte;

            SP_CALL(spMemoryHostAlloc(&buffer, total_size_in_byte));
        }

        SP_CALL(spMemoryCopy(buffer, sp->m_attrs_[i].data, total_size_in_byte));

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

    SP_CALL(spMemoryHostFree(&buffer));

    SP_CALL(spIOStreamOpen(os, curr_path));

//    SP_CALL(spParticleCoordinateGlobalToLocal(sp));

    return SP_SUCCESS;

}

int spParticleRead(struct spParticle_s *sp, spIOStream *os, const char *url, int flag)
{
    if (sp == NULL) { return SP_FAILED; }
    UNIMPLEMENTED;

    return SP_UNIMPLEMENTED;
}

int
spParticleDiagnose(spParticle const *sp, struct spIOStream_s *os, char const *path, int flag)
{
    if (sp == NULL) { return SP_FAILED; }


    char curr_path[2048];

    char new_path[2048];

    strcpy(new_path, path);

    new_path[strlen(path)] = '/';

    new_path[strlen(path) + 1] = '\0';

    SP_CALL(spIOStreamPWD(os, curr_path));

    SP_CALL(spIOStreamOpen(os, new_path));

    SP_CALL(spFieldWrite(sp->bucket_start, os, "bucket_start", flag));

    SP_CALL(spFieldWrite(sp->bucket_count, os, "bucket_count", flag));

    SP_CALL(spIOStreamOpen(os, curr_path));

    return SP_SUCCESS;
}
