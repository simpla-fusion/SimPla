//
// Created by salmon on 16-7-20.
//
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>


#include "sp_lite_def.h"
#include "spParallel.h"

#include "spMesh.h"
#include "spPage.h"
#include "spParticle.h"

typedef struct spParticleData_s { void *attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR]; } spParticleData;

struct spParticleAttrEntity_s
{
    int type_tag;
    size_type size_in_byte;
    size_type offset;
    char name[255];
    void *data;
};
#define MAX_NUM_OF_NEIGHBOUR 27

struct spParticleSyncStatus_s
{
    int num_of_neighbour;
    int num_reqs;
    MPI_Request requests[MAX_NUM_OF_NEIGHBOUR * SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    int num_of_recv;
    int *offset_recv;

};
struct spParticle_s
{
    SP_OBJECT_HEAD

    Real mass;
    Real charge;

    struct spMesh_s const *m;
    int iform;

    size_type max_num_of_entities;
    int num_of_attrs;
    struct spParticleAttrEntity_s attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR];
    struct spParticleData_s *m_data_device_; // DEVICE

    size_type max_number_of_pages;
    struct MeshEntityId *m_ids_;
    struct spParticlePage_s *m_pages_;
    struct spParticlePage_s **m_page_pool_;  // DEVICE
    struct spParticlePage_s **m_buckets_;  // DEVICE

    struct spParticleSyncStatus_s sync_reqs;
};

void spParticleCreate(const spMesh *mesh, spParticle **sp)
{
    *sp = (spParticle *) malloc(sizeof(spParticle));

    (*sp)->id = spMPIGenerateObjectId();
    (*sp)->m = mesh;
    (*sp)->iform = VERTEX;
    (*sp)->num_of_attrs = 0;
    (*sp)->m_page_pool_ = NULL;
    (*sp)->m_pages_ = NULL;
    (*sp)->m_buckets_ = NULL;

    ADD_PARTICLE_ATTRIBUTE((*sp), int64_t, flag);
    ADD_PARTICLE_ATTRIBUTE((*sp), Real, rx);
    ADD_PARTICLE_ATTRIBUTE((*sp), Real, ry);
    ADD_PARTICLE_ATTRIBUTE((*sp), Real, rz);

}

void spParticleDestroy(spParticle **sp)
{

    for (int i = 0; i < (*sp)->num_of_attrs; ++i) { spParallelDeviceFree(&((*sp)->attrs[i].data)); }

    spParallelDeviceFree((void **) &((*sp)->m_data_device_));
    spParallelDeviceFree((void **) &((*sp)->m_buckets_));
    spParallelDeviceFree((void **) &((*sp)->m_page_pool_));
    spParallelDeviceFree((void **) &((*sp)->m_pages_));

    free(*sp);
    *sp = NULL;
}

void spParticleDeploy(spParticle *sp, size_type PIC)
{
    size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, sp->iform/*volume*/);

    size_type num_page_per_cell = PIC * 2 / SP_NUMBER_OF_ENTITIES_IN_PAGE + 1;

    sp->max_number_of_pages = number_of_cell * num_page_per_cell;

    void *t_data[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    sp->max_num_of_entities = sp->max_number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;
    {

        for (int i = 0; i < sp->num_of_attrs; ++i)
        {
            spParallelDeviceMalloc(&(sp->attrs[i].data), sp->attrs[i].size_in_byte * sp->max_num_of_entities);
            spParallelMemset((sp->attrs[i].data), 0, sp->attrs[i].size_in_byte * sp->max_num_of_entities);
            t_data[i] = sp->attrs[i].data;
        }

        spParallelDeviceMalloc((void **) (&(sp->m_data_device_)), sizeof(spParticleData));

        spParallelMemcpy((void *) (sp->m_data_device_), (t_data), sizeof(void *) * sp->num_of_attrs);
    }

    spParallelDeviceMalloc((void **) (&(sp->m_pages_)), sp->max_number_of_pages * sizeof(spParticlePage));

    spParallelDeviceMalloc((void **) (&(sp->m_page_pool_)), sizeof(spPage *));

    spParallelMemcpy((void *) (sp->m_page_pool_), (void const *) &(sp->m_pages_), sizeof(spPage *));

    spParallelDeviceMalloc((void **) (&(sp->m_buckets_)), sizeof(spPage *) * number_of_cell);

    spParallelMemset((void *) ((sp->m_buckets_)), 0x0, sizeof(spPage *) * number_of_cell);


//    LOAD_KERNEL(spParticleDeployKernel,
//                sizeType2Dim3(spMeshGetShape(sp->m)), NUMBER_OF_THREADS_PER_BLOCK,
//                sp->m_buckets_,
//                sp->m_pages_,
//                sp->max_number_of_pages);

    spParallelDeviceSync();        //wait for iteration to finish

    DONE
}

spParticlePage **spParticleBuckets(spParticle *sp) { return sp->m_buckets_; };

spParticlePage **spParticlePagePool(spParticle *sp) { return sp->m_page_pool_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

void spParticleAddAttribute(struct spParticle_s *pg,
                            char const *name,
                            int type_tag,
                            size_type size_in_byte,
                            size_type offset)
{
    strcpy(pg->attrs[pg->num_of_attrs].name, name);
    pg->attrs[pg->num_of_attrs].type_tag = type_tag;
    pg->attrs[pg->num_of_attrs].size_in_byte = size_in_byte;
    if (offset == (0ul - 1))
    {
        if (pg->num_of_attrs == 0) { offset = 0; }
        else
        {
            offset = (pg->attrs[pg->num_of_attrs - 1].offset
                + pg->attrs[pg->num_of_attrs - 1].size_in_byte);
        }
    }
    pg->attrs[pg->num_of_attrs].offset = offset;
    ++pg->num_of_attrs;
    assert(pg->num_of_attrs < SP_MAX_NUMBER_OF_PARTICLE_ATTR);
}

void *spParticleAttributeData(struct spParticle_s *pg, int i)
{
    return i < pg->num_of_attrs ? pg->attrs[i].data : NULL;
};

void **spParticleAttributeDeviceData(struct spParticle_s *pg)
{
    return (void **) (pg->m_data_device_->attrs);
};

int spParticleGetibuteTypeTag(struct spParticle_s *pg, int i)
{
    return i < pg->num_of_attrs ? pg->attrs[i].type_tag : 0;
};

size_type spParticleAttibuteSizeInByte(struct spParticle_s *pg, int i)
{
    return i < pg->num_of_attrs ? pg->attrs[i].size_in_byte : 0;
}

void spParticleAttributeName(struct spParticle_s *pg, int i, char *name)
{
    if (i < pg->num_of_attrs) { strcpy(name, pg->attrs[i].name); }
}

void spMPIDataTypeCreate(int count, int *array_of_displacements, int type_tag, MPI_Datatype *new_type)
{
    MPI_Datatype ele_type = MPI_BYTE;
    switch (type_tag)
    {
        case SP_TYPE_float:
            ele_type = MPI_FLOAT;
            break;
        case SP_TYPE_double:
            ele_type = MPI_DOUBLE;
            break;

        case SP_TYPE_int:
            ele_type = MPI_INT;
            break;

        case SP_TYPE_long:
            ele_type = MPI_LONG;
            break;
        case SP_TYPE_int64_t:
            ele_type = MPI_INT64_T;
            break;
        default:
            break;
    }

    MPI_ERROR(MPI_Type_create_indexed_block(count,
                                            SP_NUMBER_OF_ENTITIES_IN_PAGE,
                                            array_of_displacements,
                                            ele_type,
                                            new_type));
    MPI_ERROR(MPI_Type_commit(new_type));
}

void spParticleSyncStart(spParticle *sp)
{
    int num_of_pages_send[MAX_NUM_OF_NEIGHBOUR];
    int num_of_pages_recv[MAX_NUM_OF_NEIGHBOUR];

    int *page_offset_send[MAX_NUM_OF_NEIGHBOUR];
    int *page_offset_recv[MAX_NUM_OF_NEIGHBOUR];

    // sync number of pages
    int num_reqs = 0;

    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
    {

        page_offset_send[i] = NULL;

        num_of_pages_recv[i] = 0;

        num_of_pages_send[i] = 0;


        size_type lower[3], upper[3];

        int offset[3];

        if (spMeshGetDomain(sp->m, i, lower, upper, offset) == 0) { continue; }
//        if (spMPIGetNeighbour(offset) == spMPIGetRank()) { continue; }

        int dest = 0, send_tag = 0, recv_tag;

        spMPIMakeSendRecvTag(sp->id, offset, &dest, &send_tag, &recv_tag);

//        num_of_pages_send[i] = (int) spParticlePageExpand(sp, lower, upper, 1024, &(page_offset_send[i]));


        MPI_ERROR(MPI_Isend(&(num_of_pages_send[i]),
                            1,
                            MPI_INT,
                            dest,
                            send_tag,
                            spMPIComm(),
                            &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
        ++(num_reqs);

        MPI_ERROR(MPI_Irecv(&(num_of_pages_recv[i]),
                            1,
                            MPI_INT,
                            dest,
                            recv_tag,
                            spMPIComm(),
                            &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
        ++(num_reqs);

    }

    MPI_ERROR(MPI_Waitall(sp->sync_reqs.num_reqs, sp->sync_reqs.requests, MPI_STATUS_IGNORE));


    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
    {
        if (num_of_pages_recv[i] <= 1) { continue; }

        spParallelHostMalloc((void **) &(page_offset_recv[i]), sizeof(int) * num_of_pages_recv[i]);

    }

    sp->sync_reqs.num_reqs = 0;

    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
    {

        size_type lower[3], upper[3];
        int offset[3];

        if (spMeshGetDomain(sp->m, i, lower, upper, offset) == 0) { continue; }
//        if (spMPIGetNeighbour(offset) == spMPIGetRank()) { continue; }
        int dest = 0, send_tag = 0, recv_tag;

        spMPIMakeSendRecvTag(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR, offset, &dest, &send_tag, &recv_tag);


        for (int s = 0; s < sp->num_of_attrs; ++s)
        {
            int tag = (int) (sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR) + s;
            spMPIMakeSendRecvTag(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR + s + 1, offset, &dest, &tag, &tag);

            MPI_Datatype send_datatype;

            spMPIDataTypeCreate(num_of_pages_send[i],
                                page_offset_send[i],
                                sp->attrs[s].type_tag,
                                &send_datatype);


            MPI_ERROR(MPI_Isend(
                sp->attrs[s].data,
                1,
                send_datatype,
                dest,
                tag,
                spMPIComm(),
                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));

            MPI_ERROR(MPI_Type_free(&send_datatype));

            ++sp->sync_reqs.num_reqs;


            MPI_Datatype recv_datatype;

//            spMPIDataTypeCreate(num_of_pages_recv,
//                                page_offset_recv,
//                                sp->attrs[s].type_tag,
//                                &recv_datatype);

            MPI_ERROR(MPI_Irecv(
                sp->attrs[s].data,
                1,
                recv_datatype,
                MPI_ANY_SOURCE,
                tag,
                spMPIComm(),
                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));

            ++sp->sync_reqs.num_reqs;

            MPI_ERROR(MPI_Type_free(&recv_datatype));

        }
    }
}

void spParticleSyncEnd(spParticle *sp)
{
    MPI_Waitall(sp->sync_reqs.num_reqs, sp->sync_reqs.requests, MPI_STATUS_IGNORE);

}

void spParticleSync(spParticle *sp)
{
    spParticleSyncStart(sp);
    spParticleSyncEnd(sp);
}



/*****************************************************************************************/
/*  2016-07-10 Salmon
 *  TODO
 *   1. page counting need optimize
 *   2. parallel write incorrect, need calculate global offset (file dataspace) before write
 *
 */
void spParticleWrite(spParticle const *sp, spIOStream *os, const char name[], int flag)
{

    char curr_path[2048];
    char new_path[2048];
    strcpy(new_path, name);
    new_path[strlen(name)] = '/';
    new_path[strlen(name) + 1] = '\0';


    spIOStreamPWD(os, curr_path);
    spIOStreamOpen(os, new_path);


    int *page_offset_host = NULL;

    size_type lower[3], upper[3];

    spMeshGetDomain(sp->m, SP_DOMAIN_CENTER, lower, upper, NULL);

    size_type num_of_pages = 0;// spParticlePageExpand(sp, lower, upper, sp->max_number_of_pages, &page_offset_host);


    if (num_of_pages > 0)
    {
        size_type num_of_entities = (size_type) (SP_NUMBER_OF_ENTITIES_IN_PAGE * num_of_pages);

        size_type count[2] = {num_of_pages, SP_NUMBER_OF_ENTITIES_IN_PAGE};

        {

            MeshEntityId *page_id_host = NULL;

            spParallelHostMalloc((void **) &page_id_host, num_of_pages * sizeof(MeshEntityId));

//            spParallelMemcpyIndexedBlock(page_id_host,
//                                         sp->m_ids_,
//                                         (int) num_of_pages,
//                                         sizeof(MeshEntityId),
//                                         page_offset_host);


            spIOStreamWriteSimple(os,
                                  "id",
                                  SP_TYPE_int64_t,
                                  page_id_host,
                                  1,
                                  count,
                                  NULL,
                                  NULL,
                                  NULL,
                                  NULL,
                                  SP_FILE_APPEND);

            spParallelHostFree((void **) &page_id_host);
        }

        for (int i = 0, ie = sp->num_of_attrs; i < ie; ++i)
        {
            void *write_buffer;

            size_type page_size_in_byte = sp->attrs[i].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;

            spParallelHostMalloc(&write_buffer, num_of_entities * sp->attrs[i].size_in_byte);

            for (int j = 0; j < num_of_pages; ++j)
            {
                spParallelMemcpy((byte_type *) (write_buffer) + j * page_size_in_byte,
                                 (byte_type *) (sp->attrs[i].data) + j * page_size_in_byte,
                                 page_size_in_byte);
            }


            spIOStreamWriteSimple(os,
                                  sp->attrs[i].name,
                                  sp->attrs[i].type_tag,
                                  write_buffer,
                                  2,
                                  count,
                                  NULL,
                                  NULL,
                                  NULL,
                                  NULL,
                                  SP_FILE_APPEND);

            spParallelHostFree(&write_buffer);

        }
    }
    spParallelHostFree((void **) &page_offset_host);

    spIOStreamOpen(os, curr_path);

}
void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag)
{

}

