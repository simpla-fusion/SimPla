//
// Created by salmon on 16-7-20.
//
#include "sp_lite_def.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>


#include "spParallel.h"
#include "spIO.h"
#include "spMesh.h"
#include "spObject.h"
#include "spParticle.h"


struct spParticleAttrEntity_s
{
    int type_tag;
    size_type size_in_byte;
    size_type offset;
    char name[255];
};
typedef struct spParticleFiber_s
{
    SP_PARTICLE_HEAD
    byte_type __other[];
} spParticleFiber;

typedef struct spParticleLinkNode_s
{
    SP_PAGE_HEAD(struct spParticleLink_s)
    spParticleFiber *data;
} spParticleLinkNode;

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

    struct spParticleAttrEntity_s m_attrs_[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    spParticleFiber *m_data_root_;// DEVICE

    size_type m_size_of_fiber_;
    size_type m_length_of_fiber_;


    MPI_Datatype m_fiber_mpi_type_;
    hid_t m_fiber_hdf5_type;

};

int spParticleCreate(const spMesh *mesh, spParticle **sp)
{
    spParallelHostAlloc(sp, sizeof(spParticle));

    (*sp)->id = spMPIGenerateObjectId();
    (*sp)->m = mesh;
    (*sp)->iform = VOLUME;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->m_fiber_mpi_type_ = MPI_DATATYPE_NULL;
    (*sp)->m_fiber_hdf5_type = H5T_NO_CLASS;

    return SP_SUCCESS;

}

int spParticleDeploy(spParticle *sp, size_type PIC)
{
    size_type number_of_cell = spMeshNumberOfEntity(sp->m, SP_DOMAIN_ALL, sp->iform);
    sp->m_length_of_fiber_ = (PIC * 2);

    spParallelDeviceAlloc((void **) &(sp->m_data_root_), sp->m_size_of_fiber_ * number_of_cell);

    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (*sp != NULL)
    {
        spParallelDeviceFree((void **) &((*sp)->m_data_root_));

        if ((*sp)->m_fiber_mpi_type_ != MPI_DATATYPE_NULL) { MPI_Type_free(&((*sp)->m_fiber_mpi_type_)); }
        if ((*sp)->m_fiber_hdf5_type != H5T_NO_CLASS) { H5Tclose((*sp)->m_fiber_hdf5_type); }
        spParallelHostFree(sp);

    }
    return SP_SUCCESS;
}

spParticleFiber *spParticleDataRoot(spParticle *sp) { return sp->m_data_root_; };


size_type spParticleNumOfEntitiesInPage(spParticle const *sp) { return sp->m_length_of_fiber_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

Real spParticleMass(spParticle const *sp) { return sp->mass; }

Real spParticleCharge(spParticle const *sp) { return sp->charge; }

void spParticleSizeOfEntity(struct spParticle_s *sp, size_type size_in_byte)
{
    sp->m_size_of_fiber_ = size_in_byte;
}

int spParticleAddAttribute(struct spParticle_s *pg, char const *name, int type_tag, size_type size_in_byte,
                           size_type offset)
{
    strcpy(pg->m_attrs_[pg->m_num_of_attrs_].name, name);

    pg->m_attrs_[pg->m_num_of_attrs_].type_tag = type_tag;

    pg->m_attrs_[pg->m_num_of_attrs_].size_in_byte = size_in_byte;

    pg->m_attrs_[pg->m_num_of_attrs_].offset = offset;

    ++pg->m_num_of_attrs_;

    assert(pg->m_num_of_attrs_ < SP_MAX_NUMBER_OF_PARTICLE_ATTR);

    return SP_SUCCESS;
}

int spParticleNumberOfAttributes(struct spParticle_s const *sp) { return sp->m_num_of_attrs_; }

/**
 *
 * @param sp
 */
void spParticleSync(spParticle *sp)
{
    int ndims = 3;

    size_type count[ndims], start[ndims], dims[ndims];
    spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, start, count, dims, NULL);

    for (int i = 0; i < 3; ++i) { count[i] -= start[i]; }

    spMPIUpdateNdArrayHalo(sp->m_data_root_,
                           ndims, dims, start, NULL, count, NULL, sp->m_fiber_mpi_type_, spMPIComm());


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
    spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, start, count, dims, NULL);
    for (int i = 0; i < 3; ++i) { count[i] -= start[i]; }

    void *buffer;
    spParallelHostAlloc(&buffer, sp->m_size_of_fiber_ * num_of_cell);
    spParallelMemcpy(buffer, sp->m_data_root_, sp->m_size_of_fiber_ * num_of_cell);

    spIOWriteSimple(os, name, sp->m_fiber_hdf5_type, buffer,
                    ndims + 1, dims, start, NULL, count, NULL, flag);

    spParallelHostFree(&buffer);


}

void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag)
{

}

