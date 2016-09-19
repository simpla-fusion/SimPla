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
#include "spAlogorithm.h"

#include "spMesh.h"
#include "spField.h"
#include "spParticle.h"
#include "detail/spParticle.impl.h"


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

    unsigned int m_pic_;

    size_type m_max_hash_;

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

size_type spParticleSize(spParticle const *sp) { return sp->m_num_of_particle_; };

int spParticleResize(spParticle *sp, size_type s)
{
    if (sp == NULL) { return SP_FAILED; }
    sp->m_num_of_particle_ = s;

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
    if (current_data != NULL) { *current_data = sp->m_current_data_; }
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


int spParticleCreate(spParticle **sp, const spMesh *mesh)
{


    SP_CALL(spMeshAttributeCreate((spMeshAttribute **) sp, sizeof(spParticle), mesh, VOLUME));
    (*sp)->m_num_of_particle_ = 0;
    (*sp)->m_max_num_of_particle_ = 0;
    (*sp)->m_pic_ = 0;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->charge = 1;
    (*sp)->mass = 1;
    (*sp)->is_deployed = SP_FALSE;
    (*sp)->step_count = 0;
    (*sp)->defragment_freq = (size_type) -1;
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

    sp->m_num_of_particle_ = spMeshGetNumberOfEntities(m, SP_DOMAIN_CENTER, iform) * sp->m_pic_;

    sp->m_max_num_of_particle_ = spMeshGetNumberOfEntities(m, SP_DOMAIN_ALL, iform) * sp->m_pic_ * 5 / 4;


    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        SP_CALL(spMemDeviceAlloc(&(sp->m_attrs_[i].data),
                                 spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * sp->m_max_num_of_particle_));
    }
    void *d[spParticleGetNumberOfAttributes(sp)];
    SP_CALL(spParticleGetAllAttributeData(sp, d));
    SP_CALL(spMemDeviceAlloc((void **) &(sp->m_current_data_),
                             spParticleGetNumberOfAttributes(sp) * sizeof(void *)));
    SP_CALL(spMemCopy(sp->m_current_data_, d,
                      spParticleGetNumberOfAttributes(sp) * sizeof(void *)));

    /* Deploy buckets */

    SP_CALL(spFieldCreate(&(sp->bucket_start), m, VOLUME, SP_TYPE_size_type));
    SP_CALL(spFieldCreate(&(sp->bucket_count), m, VOLUME, SP_TYPE_size_type));
    SP_CALL(spFieldClear(sp->bucket_start));
    SP_CALL(spFieldClear(sp->bucket_count));

    SP_CALL(spMemDeviceAlloc((void **) &(sp->sorted_idx), sp->m_max_num_of_particle_ * sizeof(size_type)));
    SP_CALL(spMemDeviceAlloc((void **) &(sp->cell_hash), sp->m_max_num_of_particle_ * sizeof(size_type)));
    sp->is_deployed = SP_TRUE;
    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (sp == NULL || *sp == NULL) { return SP_SUCCESS; }


    SP_CALL(spFieldDestroy(&(*sp)->bucket_start));
    SP_CALL(spFieldDestroy(&(*sp)->bucket_count));

    SP_CALL(spMemDeviceFree((void **) &((*sp)->sorted_idx)));
    SP_CALL(spMemDeviceFree((void **) &((*sp)->cell_hash)));

    for (int i = 0; i < (*sp)->m_num_of_attrs_; ++i) {SP_CALL(spMemDeviceFree(&((*sp)->m_attrs_[i].data))); }

    SP_CALL(spMemDeviceFree((void **) &((*sp)->m_current_data_)));
    SP_CALL(spMeshAttributeDestroy((spMeshAttribute **) sp));
    return SP_SUCCESS;
}

int spParticleInitialize(spParticle *sp, int const *dist_types)
{
    if (sp == NULL) { return SP_FAILED; }


    SP_CALL(spParticleDeploy(sp));

    /* Initialize particles*/

    SP_CALL(spMemSet(sp->cell_hash, -1, spParticleCapacity(sp) * sizeof(size_type)));

    void *data[spParticleGetNumberOfAttributes(sp)];

    SP_CALL(spParticleGetAllAttributeData(sp, data));

    size_type offset = 0, total = spParticleSize(sp);

    SP_CALL(spMPIPrefixSum(&offset, &total));

    SP_CALL(spParticleInitialize_device((Real **) (data), 6, dist_types, spParticleSize(sp), offset));

    SP_CALL(spParticleInitializeBucket_device(sp));

    return SP_SUCCESS;

}

/**  ID  and sort @{*/

int spParticleGetBucket(spParticle *sp, size_type **start_pos, size_type **end_pos, size_type **sorted_idx,
                        size_type **cell_hash)
{
    if (sp == NULL) { return SP_FAILED; }

    if (start_pos != NULL) { *start_pos = spFieldData(sp->bucket_start); }

    if (end_pos != NULL) { *end_pos = spFieldData(sp->bucket_count); }

    if (sorted_idx != NULL) { *sorted_idx = sp->sorted_idx; }

    if (cell_hash != NULL) { *cell_hash = sp->cell_hash; }

    return SP_SUCCESS;
}

int spParticleDefragment(spParticle *sp)
{
    if (sp == NULL) { return SP_FAILED; }


    size_type numParticles = spParticleSize(sp);

    size_type maxNumParticles = spParticleCapacity(sp);

    size_type *start_pos, *end_pos, *sorted_idx;

    SP_CALL(spParticleGetBucket(sp, &start_pos, &end_pos, &sorted_idx, NULL));

    Real *buffer = NULL;

    SP_CALL(spMemDeviceAlloc((void **) &buffer, sizeof(Real) * spParticleCapacity(sp)));

    for (int i = 0; i < spParticleGetNumberOfAttributes(sp); ++i)
    {
        Real *dest = buffer;

        buffer = (Real *) spParticleGetAttributeData(sp, i);

        SP_CALL(spMemoryIndirectCopy(dest, buffer, numParticles, maxNumParticles, sorted_idx));

        SP_CALL(spParticleSetAttributeData(sp, i, dest));
    }

    SP_CALL(spMemDeviceFree((void **) &buffer));

    SP_CALL(spFillSeqInt(sorted_idx, maxNumParticles, 0, 1));

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

    SP_CALL(spParticleBuildBucket_device(sp));

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

    int iform = spMeshAttributeGetForm((spMeshAttribute const *) sp);

    int ndims = spMeshGetNDims(m);

    /*******/

    SP_CALL(spFieldSync(sp->bucket_count));

    size_type *bucket_start_pos = NULL, *bucket_count = NULL, *sorted_idx = NULL;

    SP_CALL(spFieldCopyToHost((void **) &bucket_start_pos, sp->bucket_start));

    SP_CALL(spFieldCopyToHost((void **) &bucket_count, sp->bucket_count));

    SP_CALL(spMemHostAlloc((void **) &sorted_idx, sp->m_num_of_particle_ * sizeof(size_type)));

    SP_CALL(spMemCopy((void *) sorted_idx, sp->sorted_idx, sp->m_num_of_particle_ * sizeof(size_type)));


    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_end[ndims + 1];
    size_type l_strides[ndims + 1];
    size_type l_count[ndims + 1];

    SP_CALL(spMeshGetLocalDims(m, l_dims));

    SP_CALL(spMeshGetStrides(m, l_strides));

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, l_start, l_end, l_count));


    size_type p_tail = spParticleSize(sp);

    for (int i = 0; i < l_dims[0]; ++i)
        for (int j = 0; j < l_dims[1]; ++j)
            for (int k = 0; k < l_dims[2]; ++k)
            {
                if ((i >= l_start[0] && i < l_end[0]) &&
                    (j >= l_start[1] && j < l_end[1]) &&
                    (k >= l_start[2] && k < l_end[2])) { continue; }

                size_type s = i * l_strides[0] + j * l_strides[1] + k * l_strides[2];

                bucket_start_pos[s] = p_tail;

                p_tail += bucket_count[s];
            }
    SP_CALL(spParticleResize(sp, p_tail));


    SP_CALL(spFieldCopyToDevice(sp->bucket_start, bucket_start_pos));

    /*******/

    SP_CALL(spMeshGetLocalDims(m, l_dims));

    SP_CALL(spMeshGetDomain(m, SP_DOMAIN_CENTER, l_start, l_end, l_count));

    /* MPI COMM Start */

    void *d[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    SP_CALL(spParticleGetAllAttributeData(sp, d));


    spMPICartUpdater *updater;

    SP_CALL(spMPICartUpdaterCreate(&updater,
                                   spMPIComm(),
                                   SP_TYPE_Real,
                                   0,
                                   ndims,
                                   l_dims,
                                   l_start,
                                   NULL,
                                   l_count,
                                   NULL,
                                   bucket_start_pos,
                                   bucket_count,
                                   sorted_idx));


    SP_CALL(spMPICartUpdateAll(updater, spParticleGetNumberOfAttributes(sp), d));

    SP_CALL(spMPICartUpdaterDestroy(&updater));

    /* MPI COMM End*/

    SP_CALL(spMemHostFree((void **) &bucket_start_pos));

    SP_CALL(spMemHostFree((void **) &bucket_count));

    SP_CALL(spMemHostFree((void **) &sorted_idx));

    return SP_SUCCESS;

}

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
            SP_CALL(spMemHostFree(&buffer));

            total_size_in_byte = new_total_size_in_byte;

            SP_CALL(spMemHostAlloc(&buffer, total_size_in_byte));
        }

        SP_CALL(spMemCopy(buffer, sp->m_attrs_[i].data, total_size_in_byte));

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

    SP_CALL(spMemHostFree(&buffer));

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
