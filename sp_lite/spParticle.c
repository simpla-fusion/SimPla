//
// Created by salmon on 16-7-20.
//
#include "sp_lite_def.h"
#include "../src/sp_capi.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#include "spObject.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spParallel.h"

#include "spRandom.h"

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
    SP_MESH_ATTR_HEAD

    Real mass;
    Real charge;

    int m_num_of_attrs_;

    size_type m_max_fiber_length_;

    spParticleAttrEntity m_attrs_[SP_MAX_NUMBER_OF_PARTICLE_ATTR];
    void *m_data_root_[SP_MAX_NUMBER_OF_PARTICLE_ATTR];// DEVICE
    void **m_data_root_device_;// DEVICE



};

int spParticleCreate(spParticle **sp, const spMesh *mesh)
{
    SP_CALL(spMeshAttrCreate((spMeshAttr **) sp, sizeof(spParticle), mesh, VOLUME));

    (*sp)->m_max_fiber_length_ = SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE;
    (*sp)->m_num_of_attrs_ = 0;
    (*sp)->m_data_root_device_ = NULL;

    return SP_SUCCESS;

}

int spParticleDestroy(spParticle **sp)
{
    if (*sp != NULL)
    {
        spParallelDeviceFree((void **) &((*sp)->m_data_root_device_));

        for (int i = 0; i < (*sp)->m_num_of_attrs_; ++i)
        {
            spParallelDeviceFree(&((*sp)->m_attrs_[i].data));
            SP_CALL(spDataTypeDestroy(&((*sp)->m_attrs_[i].data_type)));
        }

    }
    SP_CALL(spMeshAttrDestroy((spMeshAttr **) sp));


    return SP_SUCCESS;
}

int spParticleAddAttribute(spParticle *sp, char const name[], int tag, size_type size, size_type offset)
{
    SP_CALL(spDataTypeCreate(&(sp->m_attrs_[sp->m_num_of_attrs_].data_type), tag, size));

    sp->m_attrs_[sp->m_num_of_attrs_].offset = offset;
    strcpy(sp->m_attrs_[sp->m_num_of_attrs_].name, name);
    sp->m_attrs_[sp->m_num_of_attrs_].data = NULL;

    ++(sp->m_num_of_attrs_);
}

int spParticleGetNumberOfAttributes(spParticle const *sp)
{
    return sp->m_num_of_attrs_;
}

char const *spParticleGetAttributeName(spParticle *sp, int i) { return sp->m_attrs_[i].name; };

size_type spParticleGetAttributeTypeSizeInByte(spParticle *sp, int i)
{
    return spDataTypeSizeInByte(sp->m_attrs_[i].data_type);
};

void *spParticleGetAttributeData(spParticle *sp, int i) { return sp->m_attrs_[i].data; }

int spParticleDeploy(spParticle *sp)
{
    size_type number_of_entities = spParticleGetNumberOfEntities(sp);

    assert (sp->m_max_fiber_length_ > 0);


    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        spParallelDeviceAlloc(&(sp->m_attrs_[i].data),
                              spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * number_of_entities);


        sp->m_data_root_[i] = sp->m_attrs_[i].data;
    }

    spParallelDeviceAlloc((void **) &(sp->m_data_root_device_), sizeof(void *) * sp->m_num_of_attrs_);
    spParallelMemcpy((void *) (sp->m_data_root_device_), sp->m_data_root_, sizeof(void *) * sp->m_num_of_attrs_);
    return SP_SUCCESS;
}

size_type spParticleGetNumberOfEntities(spParticle const *sp)
{
    spMesh const *m = spMeshAttrMesh((spMeshAttr *) (sp));
    size_type s = spParticleGetMaxFiberLength(sp);
    size_type n = spMeshNumberOfEntity(m, SP_DOMAIN_ALL, spMeshAttrForm((spMeshAttr *) (sp)));
    return n * spParticleGetMaxFiberLength(sp);

}

int spParticleInitialize(spParticle *sp, size_type num_of_pic, int const *dist_types)
{
    spMesh const *m = spMeshAttrMesh((spMeshAttr *) sp);

    int iform = spMeshAttrForm((spMeshAttr *) sp);

    size_type max_number_of_entities = spParticleGetNumberOfEntities(sp);

    int num_of_dimensions = spParticleGetNumberOfAttributes(sp);

    int l_dist_types[num_of_dimensions];
    for (int i = 0; i < num_of_dimensions - 1; ++i)
    {
        l_dist_types[i] = dist_types == NULL ? SP_RAND_UNIFORM : dist_types[i];
    }

    spParticleFiber *data = (spParticleFiber *) spParticleGetData(sp);

    SP_CALL(spParallelMemset(data->id, 0, max_number_of_entities * sizeof(int)));

    size_type x_min[3], x_max[3], strides[3];
    spMeshLocalDomain2(m, SP_DOMAIN_CENTER, x_min, x_max, strides);
    strides[0] *= spParticleGetMaxFiberLength(sp);
    strides[1] *= spParticleGetMaxFiberLength(sp);
    strides[2] *= spParticleGetMaxFiberLength(sp);


    size_type offset = 0;// spMeshNumberOfEntity(m, SP_DOMAIN_CENTER, iform) * num_of_pic;

//        spParallelScan(&offset);
    spRandomGenerator *sp_gen;
    spRandomGeneratorCreate(&sp_gen, SP_RAND_GEN_SOBOL, num_of_dimensions, offset);

    spRandomDistributionInCell(sp_gen, l_dist_types,
                               (Real **) (spParticleGetDeviceData(sp) + 1),
                               x_min, x_max, strides, max_number_of_entities);

    spRandomGeneratorDestroy(&sp_gen);


}

int spParticleSetPIC(spParticle *sp, size_type pic)
{
    sp->m_max_fiber_length_ =
            (3 * pic / SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE / 2 + 1) * SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE;

    return SP_SUCCESS;
}

size_type spParticleGetMaxFiberLength(const spParticle *sp) { return sp->m_max_fiber_length_; }

void **spParticleGetDeviceData(spParticle *sp) { return sp->m_data_root_device_; };

void **spParticleGetData(spParticle *sp) { return sp->m_data_root_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

int spParticleSetMass(spParticle *sp, Real m)
{
    sp->mass = m;
    return SP_SUCCESS;
}

int spParticleSetCharge(spParticle *sp, Real e)
{
    sp->charge = e;
    return SP_SUCCESS;
}

Real spParticleGetMass(spParticle const *sp) { return sp->mass; }

Real spParticleGetCharge(spParticle const *sp) { return sp->charge; }

int spParticleNumberOfAttributes(struct spParticle_s const *sp) { return sp->m_num_of_attrs_; }

/**
 *
 * @param sp
 */
int spParticleSync(spParticle *sp)
{


    spMesh const *m = spMeshAttrMesh((spMeshAttr const *) sp);
    int iform = spMeshAttrForm((spMeshAttr const *) sp);
    int ndims = spMeshNDims(m);
    int array_ndims, mesh_start_dim;

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_count[ndims + 1];

    size_type num_of_entities = spParticleGetMaxFiberLength(sp);

    SP_CALL(spMeshArrayShape(m, SP_DOMAIN_CENTER, 1, &num_of_entities,
                             &array_ndims, &mesh_start_dim, NULL, NULL, l_dims, l_start, l_count, SP_FALSE));


    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        SP_CALL(spParallelUpdateNdArrayHalo(sp->m_attrs_[i].data, sp->m_attrs_[i].data_type,
                                            array_ndims, l_dims, l_start, NULL, l_count, NULL, 0));
    }
    return SP_SUCCESS;

}


int
spParticleWrite(spParticle const *sp, spIOStream *os, const char *name, int flag)
{
    if (sp == NULL) { return SP_FAILED; }

    char curr_path[2048];
    char new_path[2048];
    strcpy(new_path, name);
    new_path[strlen(name)] = '/';
    new_path[strlen(name) + 1] = '\0';


    SP_CALL(spIOStreamPWD(os, curr_path));
    SP_CALL(spIOStreamOpen(os, new_path));


    spMesh const *m = spMeshAttrMesh((spMeshAttr const *) sp);

    int iform = spMeshAttrForm((spMeshAttr const *) sp);

    int ndims = spMeshNDims(m);

    int array_ndims, mesh_start_dim;

    size_type l_dims[ndims + 1];
    size_type l_start[ndims + 1];
    size_type l_count[ndims + 1];

    size_type g_dims[ndims + 1];
    size_type g_start[ndims + 1];

    size_type num_of_entities = spParticleGetMaxFiberLength(sp);

    spMeshArrayShape(m, SP_DOMAIN_CENTER, 1, &num_of_entities,
                     &array_ndims, &mesh_start_dim, g_dims, g_start, l_dims, l_start, l_count,
                     SP_FALSE);

    num_of_entities *= spMeshNumberOfEntity(m, SP_DOMAIN_ALL, iform);

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        void *buffer = NULL;

        size_type size_in_byte = spDataTypeSizeInByte(sp->m_attrs_[i].data_type) * num_of_entities;

        spParallelHostAlloc(&buffer, size_in_byte);

        spParallelMemcpy(buffer, sp->m_attrs_[i].data, size_in_byte);

        SP_CALL(spIOStreamWriteSimple(os,
                                      sp->m_attrs_[i].name,
                                      sp->m_attrs_[i].data_type,
                                      buffer,
                                      array_ndims,
                                      l_dims,
                                      l_start,
                                      NULL,
                                      l_count,
                                      NULL,
                                      g_dims,
                                      g_start,
                                      flag));

        spParallelHostFree(&buffer);
    }

    SP_CALL(spIOStreamOpen(os, curr_path));
    return SP_SUCCESS;

}

int spParticleRead(struct spParticle_s *f, spIOStream *os, const char *url, int flag)
{
    return SP_SUCCESS;
}

