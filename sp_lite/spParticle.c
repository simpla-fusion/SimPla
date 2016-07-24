//
// Created by salmon on 16-7-20.
//
#include "sp_lite_def.h"
#include "../src/sp_capi.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>


#include "spParallel.h"
#include "spMesh.h"
#include "spObject.h"
#include "spParticle.h"


#ifndef SP_MAX_NUMBER_OF_PARTICLE_ATTR
#    define SP_MAX_NUMBER_OF_PARTICLE_ATTR 16
#endif


typedef struct spParticleFiber_s
{
    SP_PARTICLE_HEAD
    byte_type __other[];
} spParticleFiber;

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
    SP_OBJECT_HEAD

    Real mass;
    Real charge;

    struct spMesh_s const *m;

    int iform;

    spParticleFiber *m_data_root_;// DEVICE

    spDataType *m_data_type_desc_;
};

int spParticleCreate(spParticle **sp, const spMesh *mesh)
{
    spParallelHostAlloc(sp, sizeof(spParticle));

    (*sp)->id = spMPIGenerateObjectId();
    (*sp)->m = mesh;
    (*sp)->iform = VOLUME;
    (*sp)->m_data_type_desc_ = NULL;
    spDataTypeCreate(&((*sp)->m_data_type_desc_), SP_TYPE_NULL);
    return SP_SUCCESS;

}

int spParticleDeploy(spParticle *sp, size_type PIC)
{
    size_type number_of_cell = spMeshNumberOfEntity(sp->m, SP_DOMAIN_ALL, sp->iform);

    spParallelDeviceAlloc((void **) &(sp->m_data_root_), spDataTypeSizeInByte(sp->m_data_type_desc_) * number_of_cell);

    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (*sp != NULL)
    {
        spParallelDeviceFree((void **) &((*sp)->m_data_root_));

        SP_CHECK_RETURN(spDataTypeDestroy(&((*sp)->m_data_type_desc_)));

        spParallelHostFree(sp);

    }
    return SP_SUCCESS;
}
spDataType *spParticleDataTypeDesc(spParticle *sp) { return sp->m_data_type_desc_; }

void *spParticleData(spParticle *sp) { return sp->m_data_root_; };

void const *spParticleDataConst(spParticle *sp) { return sp->m_data_root_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

Real spParticleMass(spParticle const *sp) { return sp->mass; }

Real spParticleCharge(spParticle const *sp) { return sp->charge; }

//void spParticleSizeOfEntity(struct spParticle_s *sp, size_type size_in_byte)
//{
//    sp->m_size_of_fiber_ = size_in_byte;
//}
//
//int spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag, size_type size_in_byte,
//                           size_type offset)
//{
//    strcpy(pg->m_attrs_[pg->m_num_of_attrs_].name, name);
//
//    pg->m_attrs_[pg->m_num_of_attrs_].type_tag = type_tag;
//
//    pg->m_attrs_[pg->m_num_of_attrs_].size_in_byte = size_in_byte;
//
//    pg->m_attrs_[pg->m_num_of_attrs_].offset = offset;
//
//    ++pg->m_num_of_attrs_;
//
//    assert(pg->m_num_of_attrs_ < SP_MAX_NUMBER_OF_PARTICLE_ATTR);
//
//    return SP_SUCCESS;
//}
//
//int spParticleNumberOfAttributes(struct spParticle_s const *sp) { return sp->m_num_of_attrs_; }

/**
 *
 * @param sp
 */
int spParticleSync(spParticle *sp)
{
    int ndims = 3;

    size_type count[ndims], start[ndims], dims[ndims];

    SP_CHECK_RETURN(spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, dims, start, count, NULL));

    for (int i = 0; i < 3; ++i) { count[i] -= start[i]; }

    SP_CHECK_RETURN(spParallelUpdateNdArrayHalo(sp->m_data_root_,
                                                ndims,
                                                dims,
                                                start,
                                                NULL,
                                                count,
                                                NULL,
                                                sp->m_data_type_desc_));

    return SP_SUCCESS;

}

/*****************************************************************************************
 **  2016-07-10 Salmon
 *  TODO
 *   1. page counting need optimize
 *   2. parallel write incorrect, need calculate global offset (file dataspace) before write
 *
 */
int
spParticleWrite(spParticle const *sp, spIOStream *os, const char *name, int flag)
{
    if (sp == NULL) { return SP_FAILED; }

    int ndims = 3;

    size_type num_of_cell = spMeshNumberOfEntity(sp->m, SP_DOMAIN_ALL, sp->iform);

    size_type count[ndims], start[ndims], dims[ndims];

    SP_CHECK_RETURN(spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, dims, start, count, NULL));

    for (int i = 0; i < 3; ++i) { count[i] -= start[i]; }

    void *buffer;

    size_type size_in_byte = spDataTypeSizeInByte(sp->m_data_type_desc_);

    spParallelHostAlloc(&buffer, size_in_byte * num_of_cell);

    spParallelMemcpy(buffer, sp->m_data_root_, size_in_byte * num_of_cell);

    SP_CHECK_RETURN(spIOStreamWriteSimple(os, name, sp->m_data_type_desc_, buffer,
                                          ndims, dims, start, NULL, count, NULL, flag));

    spParallelHostFree(&buffer);

    return SP_SUCCESS;

}

int spParticleRead(struct spParticle_s *f, spIOStream *os, const char *url, int flag)
{
    return SP_SUCCESS;
}

