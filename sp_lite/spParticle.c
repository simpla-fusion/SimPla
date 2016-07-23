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
    char name[255];
    void *data;
};

typedef struct spParticlePage_s
{
    SP_PAGE_HEAD(struct spParticlePage_s)
    size_type offset;
} spParticlePage;

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

    size_type m_max_num_of_pages_;

    size_type m_num_of_entities_in_page_;

    void **m_data_root_; // DEVICE

    spParticlePage *m_page_links_head_;// DEVICE

    spParticlePage **m_page_pool_;// DEVICE

    spParticlePage **m_page_base_field_;  // DEVICE  base manifold B

    int *m_page_count_;// DEVICE

};

int spParticleCreate(const spMesh *mesh, spParticle **sp)
{
    spParallelHostAlloc(sp, sizeof(spParticle));

    (*sp)->id = spMPIGenerateObjectId();
    (*sp)->m = mesh;
    (*sp)->iform = VOLUME;
    (*sp)->m_num_of_attrs_ = 0;

    return SP_SUCCESS;

}

int spParticleDeploy(spParticle *sp, size_type PIC)
{
    size_type number_of_cell = spMeshNumberOfEntity(sp->m, SP_DOMAIN_ALL, sp->iform);

    sp->m_num_of_entities_in_page_ = SP_NUMBER_OF_ENTITIES_IN_PAGE;

    sp->m_max_num_of_pages_ = number_of_cell * (PIC * 2 / sp->m_num_of_entities_in_page_ + 1);


    size_type max_num_of_entities = sp->m_num_of_entities_in_page_ * sp->m_max_num_of_pages_;

    void *data_root_host[sp->m_num_of_attrs_];

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {
        spParallelDeviceAlloc(&(sp->m_attrs_[i].data), sp->m_attrs_[i].size_in_byte * max_num_of_entities);
        data_root_host[i] = sp->m_attrs_[i].data;
    }

    spParallelDeviceAlloc((void **) &(sp->m_data_root_), sizeof(void *) * sp->m_num_of_attrs_);

    spParallelMemcpy(sp->m_data_root_, data_root_host, sizeof(void *) * sp->m_num_of_attrs_);


    spParallelDeviceAlloc((void **) &(sp->m_page_links_head_), sp->m_max_num_of_pages_ * sizeof(spParticlePage));
    {
        spParticlePage *page_host;

        spParallelHostAlloc(&page_host, sizeof(spParticlePage) * sp->m_max_num_of_pages_);

        for (int i = 0; i < sp->m_max_num_of_pages_; ++i)
        {
            page_host[i].next = sp->m_page_links_head_ + (i + 1);
            page_host[i].offset = i * sp->m_num_of_entities_in_page_;
        }

        page_host[sp->m_max_num_of_pages_].next = NULL;

        spParallelMemcpy(sp->m_page_links_head_, page_host, sizeof(spParticlePage) * sp->m_max_num_of_pages_);

        spParallelHostFree(&page_host);
    }

    spParallelDeviceAlloc((void **) (&(sp->m_page_pool_)), sizeof(spParticlePage *));

    spParallelMemcpy((void *) (sp->m_page_pool_), &(sp->m_page_links_head_), sizeof(spParticlePage *));

    spParallelDeviceAlloc((void **) (&(sp->m_page_base_field_)), sizeof(spParticlePage *) * number_of_cell);

    spParallelMemset((void *) ((sp->m_page_base_field_)), 0, sizeof(spParticlePage *) * number_of_cell);

    spParallelDeviceAlloc((void **) (&(sp->m_page_count_)), sizeof(size_type) * number_of_cell);

    spParallelMemset((void *) ((sp->m_page_count_)),
                     (int) (PIC / sp->m_num_of_entities_in_page_ + 1),
                     sizeof(size_type) * number_of_cell);

    return SP_SUCCESS;
}

int spParticleDestroy(spParticle **sp)
{
    if (*sp != NULL)
    {
        spParallelDeviceFree((void **) &((*sp)->m_data_root_));
        spParallelDeviceFree((void **) &((*sp)->m_page_links_head_));
        spParallelDeviceFree((void **) &((*sp)->m_page_pool_));
        spParallelDeviceFree((void **) &((*sp)->m_page_base_field_));
        spParallelDeviceFree((void **) &((*sp)->m_page_count_));
        spParallelHostFree(sp);
    }
    return SP_SUCCESS;
}

void **spParticleDataRoot(spParticle *sp) { return sp->m_data_root_; };

spPage **spParticleBaseField(spParticle *sp) { return (spPage **) sp->m_page_base_field_; };

spPage **spParticlePagePool(spParticle *sp) { return (spPage **) sp->m_page_pool_; };

int *spParticlePageCount(spParticle *sp) { return sp->m_page_count_; };

size_type spParticleNumOfEntitiesInPage(spParticle const *sp) { return sp->m_num_of_entities_in_page_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

Real spParticleMass(spParticle const *sp) { return sp->mass; }

Real spParticleCharge(spParticle const *sp) { return sp->charge; }

int spParticleAddAttribute(struct spParticle_s *pg,
                           char const *name,
                           int type_tag,
                           size_type size_in_byte)
{
    strcpy(pg->m_attrs_[pg->m_num_of_attrs_].name, name);

    pg->m_attrs_[pg->m_num_of_attrs_].type_tag = type_tag;

    pg->m_attrs_[pg->m_num_of_attrs_].size_in_byte = size_in_byte;

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
    spParticleResizePageLink(sp);

    int mpi_topology_ndims = spMPITopologyNDims();

    size_type *send_disps[mpi_topology_ndims * 2];

    size_type send_block_count[mpi_topology_ndims * 2];

    size_type *recv_disps[mpi_topology_ndims * 2];

    size_type recv_block_count[mpi_topology_ndims * 2];

    size_type upper[3], lower[3], shape[3];

    int ndims = 3;

    spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, lower, upper, shape, NULL);

    size_type dims[ndims], start[ndims], count[ndims];

    for (int i = 0; i < ndims; ++i)
    {
        dims[i] = shape[i];
        start[i] = (lower[i]);
        count[i] = (upper[i] - lower[i]);
    }

    size_type s_count_lower[ndims];
    size_type s_start_lower[ndims];
    size_type s_count_upper[ndims];
    size_type s_start_upper[ndims];

    size_type r_count_lower[ndims];
    size_type r_start_lower[ndims];
    size_type r_count_upper[ndims];
    size_type r_start_upper[ndims];


    for (int d = 0; d < mpi_topology_ndims; ++d)
    {

        for (int i = 0; i < ndims; ++i)
        {
            if (i < d)
            {
                s_count_lower[i] = dims[i];
                s_start_lower[i] = 0;
                s_count_upper[i] = dims[i];
                s_start_upper[i] = 0;

                r_count_lower[i] = dims[i];
                r_start_lower[i] = 0;
                r_count_upper[i] = dims[i];
                r_start_upper[i] = 0;
            }
            else if (i == d)
            {
                s_count_lower[i] = start[i];
                s_start_lower[i] = start[i];
                s_count_upper[i] = (dims[i] - count[i] - start[i]);
                s_start_upper[i] = (start[i] + count[i] - s_count_upper[i]);

                r_count_lower[i] = start[i];
                r_start_lower[i] = 0;
                r_count_upper[i] = (dims[i] - count[i] - start[i]);
                r_start_upper[i] = dims[i] - s_count_upper[i];
            }
            else
            {
                s_count_lower[i] = count[i];
                s_start_lower[i] = start[i];
                s_count_upper[i] = count[i];
                s_start_upper[i] = start[i];

                r_count_lower[i] = count[i];
                r_start_lower[i] = start[i];
                r_count_upper[i] = count[i];
                r_start_upper[i] = start[i];
            };
        }


        spParticleGetPageOffset(sp,
                                s_start_lower,
                                s_count_lower,
                                &send_block_count[2 * d + 0],
                                NULL,
                                &send_disps[2 * d + 0]);
        spParticleGetPageOffset(sp,
                                s_start_lower,
                                s_count_lower,
                                &send_block_count[2 * d + 1],
                                NULL,
                                &send_disps[2 * d + 1]);
        spParticleGetPageOffset(sp,
                                r_start_lower,
                                r_count_lower,
                                &recv_block_count[2 * d + 0],
                                NULL,
                                &recv_disps[2 * d + 0]);
        spParticleGetPageOffset(sp,
                                r_start_lower,
                                r_count_lower,
                                &recv_block_count[2 * d + 1],
                                NULL,
                                &recv_disps[2 * d + 1]);

    }


    size_type block_length = sp->m_num_of_entities_in_page_;

    MPI_Comm comm = spMPIComm();


    for (int n = 0; n < sp->m_num_of_attrs_; ++n)
    {
        MPI_Datatype ele_type;

        spMPIDataTypeCreate(sp->m_attrs_[n].type_tag, sp->m_attrs_[n].size_in_byte, &ele_type);

        spUpdateIndexedBlock(sp->m_attrs_[n].data,
                             (size_type const **) send_disps,
                             send_block_count,
                             sp->m_attrs_[n].data,
                             (size_type const **) recv_disps,
                             recv_block_count,
                             block_length,
                             ele_type,
                             comm);

        MPI_Type_free(&ele_type);

    }

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

    char curr_path[2048];
    char new_path[2048];
    strcpy(new_path, name);
    new_path[strlen(name)] = '/';
    new_path[strlen(name) + 1] = '\0';

    spIOStreamPWD(os, curr_path);
    spIOStreamOpen(os, new_path);


    size_type upper[3], lower[3], shape[3];

    int ndims = 3;

    spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, lower, upper, shape, NULL);

    size_type num_of_pages = 0;
    size_type *page_disps = NULL;
    MeshEntityId *page_ids = NULL;
    spParticleGetPageOffset((spParticle *) sp,
                            lower,
                            upper,
                            &num_of_pages,
                            &page_ids,
                            &page_disps);

    spIOWriteSimple(os,
                    "id",
                    SP_TYPE_int64_t,
                    page_ids,
                    1,
                    &num_of_pages,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    SP_FILE_APPEND);

    void *buffer[sp->m_num_of_attrs_];
    char attr_name[sp->m_num_of_attrs_][255];
    int type_tage[sp->m_num_of_attrs_];
    for (int i = 0, ie = sp->m_num_of_attrs_; i < ie; ++i)
    {

        spParallelHostAlloc(&buffer,
                            sp->m_attrs_[i].size_in_byte * sp->m_num_of_entities_in_page_ * sp->m_max_num_of_pages_);

        MPI_Datatype d_type;

        spMPIDataTypeCreate(sp->m_attrs_[i].type_tag, sp->m_attrs_[i].size_in_byte, &d_type);


        spParallelHostFree((void **) &buffer);
    }

    spIOWriteIndexedBlockSimple(os,
                                attr_name,
                                type_tage,
                                buffer,
                                sp->m_num_of_entities_in_page_ * sp->m_max_num_of_pages_,
                                num_of_pages,
                                page_disps,
                                SP_FILE_APPEND);

    spIOStreamOpen(os, curr_path);

}

void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag)
{

}

