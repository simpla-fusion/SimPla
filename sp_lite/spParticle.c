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
    SP_OBJECT_HEAD

    Real mass;
    Real charge;

    struct spMesh_s const *m;

    int iform;

    int m_num_of_attrs_;

    size_type m_max_fiber_length_;

    spParticleAttrEntity m_attrs_[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    void **m_data_root_;// DEVICE


};

int spParticleCreate(spParticle **sp, const spMesh *mesh)
{
    spParallelHostAlloc((void **) sp, sizeof(spParticle));

    (*sp)->id = spMPIGenerateObjectId();
    (*sp)->m = mesh;
    (*sp)->iform = VOLUME;
    (*sp)->m_max_fiber_length_ = SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->m_data_root_ = NULL;

    return SP_SUCCESS;

}

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset)
{
    spDataTypeCreate(&(sp->m_attrs_[sp->m_num_of_attrs_].data_type), tag, size);
    spDataTypeUpdate(sp->m_attrs_[sp->m_num_of_attrs_].data_type);

    sp->m_attrs_[sp->m_num_of_attrs_].offset = offset;
    strcpy(sp->m_attrs_[sp->m_num_of_attrs_].name, name);
    sp->m_attrs_[sp->m_num_of_attrs_].data = NULL;

    ++(sp->m_num_of_attrs_);
}
int spParticleDeploy(spParticle *sp)
{
    size_type number_of_cell = spMeshNumberOfEntity(sp->m, SP_DOMAIN_ALL, sp->iform);

    assert (sp->m_max_fiber_length_ > 0);

    void *attr_data[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        spParallelDeviceAlloc(&(sp->m_attrs_[i].data),
                              spDataTypeSizeInByte(sp->m_attrs_[i].data_type)
                                  * number_of_cell * sp->m_max_fiber_length_);
        attr_data[i] = sp->m_attrs_[i].data;
    }

    spParallelDeviceAlloc((void **) &(sp->m_data_root_), sizeof(void *) * sp->m_num_of_attrs_);
    spParallelMemcpy((void *) (sp->m_data_root_), attr_data, sizeof(void *) * sp->m_num_of_attrs_);
    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (*sp != NULL)
    {
        spParallelDeviceFree((void **) &((*sp)->m_data_root_));

        for (int i = 0; i < (*sp)->m_num_of_attrs_; ++i)
        {
            spParallelDeviceFree(&((*sp)->m_attrs_[i].data));
            spDataTypeDestroy(&((*sp)->m_attrs_[i].data_type));
        }

        spParallelHostFree(sp);

    }
    return SP_SUCCESS;
}
int spParticlePIC(spParticle *sp, size_type pic)
{
    sp->m_max_fiber_length_ =
        (2 * pic / SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE + 1) * SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE;
    return SP_SUCCESS;
}

size_type spParticleMaxFiberLength(const spParticle *sp) { return sp->m_max_fiber_length_; }

void **spParticleData(spParticle *sp) { return sp->m_data_root_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

Real spParticleMass(spParticle const *sp) { return sp->mass; }

Real spParticleCharge(spParticle const *sp) { return sp->charge; }

int spParticleNumberOfAttributes(struct spParticle_s const *sp) { return sp->m_num_of_attrs_; }

/**
 *
 * @param sp
 */
int spParticleSync(spParticle *sp)
{
    int ndims = 3;

    size_type count[ndims + 1], start[ndims + 1], dims[ndims + 1];

    SP_CHECK_RETURN(spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, dims, start, count));

    count[ndims] = spParticleMaxFiberLength(sp);

    start[ndims] = 0;

    dims[ndims] = spParticleMaxFiberLength(sp);

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {


        SP_CHECK_RETURN(spParallelUpdateNdArrayHalo(sp->m_attrs_[i].data,
                                                    sp->m_attrs_[i].data_type,
                                                    ndims + 1,
                                                    dims,
                                                    start,
                                                    NULL,
                                                    count,
                                                    NULL));
    }
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
    char curr_path[2048];
    char new_path[2048];
    strcpy(new_path, name);
    new_path[strlen(name)] = '/';
    new_path[strlen(name) + 1] = '\0';


    SP_CHECK_RETURN(spIOStreamPWD(os, curr_path));
    SP_CHECK_RETURN(spIOStreamOpen(os, new_path));


    if (sp == NULL) { return SP_FAILED; }

    int ndims = spMeshNDims(spParticleMesh(sp));

    size_type num_of_cell = spMeshNumberOfEntity(sp->m, SP_DOMAIN_ALL, sp->iform);

    size_type count[ndims + 1], start[ndims + 1], dims[ndims + 1];
    size_type g_dims[ndims + 1];
    size_type g_start[ndims + 1];

    SP_CHECK_RETURN(spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, dims, start, count));

    spMeshGlobalDomain(spParticleMesh(sp), g_dims, g_start);

    dims[ndims] = spParticleMaxFiberLength(sp);
    g_dims[ndims] = spParticleMaxFiberLength(sp);
    g_start[ndims] = 0;
    start[ndims] = 0;
    count[ndims] = spParticleMaxFiberLength(sp);

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        void *buffer = NULL;

        size_type size_in_byte = spDataTypeSizeInByte(sp->m_attrs_[i].data_type);

        size_in_byte *= spParticleMaxFiberLength(sp) * num_of_cell;

        spParallelHostAlloc(&buffer, size_in_byte);

        spParallelMemcpy(buffer, sp->m_attrs_[i].data, size_in_byte);

        SP_CHECK_RETURN(spIOStreamWriteSimple(os,
                                              sp->m_attrs_[i].name,
                                              sp->m_attrs_[i].data_type,
                                              buffer,
                                              ndims + 1,
                                              dims,
                                              start,
                                              NULL,
                                              count,
                                              NULL,
                                              g_dims,
                                              g_start,
                                              flag));
        spParallelHostFree(&buffer);
    }

    SP_CHECK_RETURN(spIOStreamOpen(os, curr_path));
    return SP_SUCCESS;

}

int spParticleRead(struct spParticle_s *f, spIOStream *os, const char *url, int flag)
{
    return SP_SUCCESS;
}

