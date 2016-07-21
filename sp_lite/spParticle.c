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
#include "spObject.h"
#include "spPage.h"
#include "spParticle.h"


struct spParticleAttrEntity_s
{
    int type_tag;
    size_type size_in_byte;
    size_type offset;
    char name[255];
    void *data;
};

struct spParticleSyncStatus_s
{
    int num_reqs;

    MPI_Request *requests;

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
    size_type *m_pages_;
    size_type **m_page_pool_;  // DEVICE
    size_type **m_buckets_;  // DEVICE

    struct spParticleSyncStatus_s sync_reqs;
};

void spParticleCreate(const spMesh *mesh, spParticle **sp)
{
//    *sp = (spParticle *) malloc(sizeof(spParticle));
//
//    (*sp)->id = spMPIGenerateObjectId();
//    (*sp)->m = mesh;
//    (*sp)->iform = VERTEX;
//    (*sp)->num_of_attrs = 0;
//    (*sp)->m_page_pool_ = NULL;
//    (*sp)->m_pages_ = NULL;
//    (*sp)->m_buckets_ = NULL;
//
//    ADD_PARTICLE_ATTRIBUTE((*sp), int64_t, flag);
//    ADD_PARTICLE_ATTRIBUTE((*sp), Real, rx);
//    ADD_PARTICLE_ATTRIBUTE((*sp), Real, ry);
//    ADD_PARTICLE_ATTRIBUTE((*sp), Real, rz);

}

void spParticleDestroy(spParticle **sp)
{

//    for (int i = 0; i < (*sp)->num_of_attrs; ++i) { spParallelDeviceFree(&((*sp)->attrs[i].data)); }
//
//    spParallelDeviceFree((void **) &((*sp)->m_data_device_));
//    spParallelDeviceFree((void **) &((*sp)->m_buckets_));
//    spParallelDeviceFree((void **) &((*sp)->m_page_pool_));
//    spParallelDeviceFree((void **) &((*sp)->m_pages_));
//
//    free(*sp);
//    *sp = NULL;
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
            spParallelDeviceAlloc(&(sp->attrs[i].data), sp->attrs[i].size_in_byte * sp->max_num_of_entities);
            spParallelMemset((sp->attrs[i].data), 0, sp->attrs[i].size_in_byte * sp->max_num_of_entities);
            t_data[i] = sp->attrs[i].data;
        }

        spParallelDeviceAlloc((void **) (&(sp->m_data_device_)), sizeof(void **));

        spParallelMemcpy((void *) (sp->m_data_device_), (t_data), sizeof(void *) * sp->num_of_attrs);
    }

    spParallelDeviceAlloc((void **) (&(sp->m_pages_)), sp->max_number_of_pages * sizeof(size_type));

    spParallelDeviceAlloc((void **) (&(sp->m_page_pool_)), sizeof(spPage *));

    spParallelMemcpy((void *) (sp->m_page_pool_), (void const *) &(sp->m_pages_), sizeof(spPage *));

    spParallelDeviceAlloc((void **) (&(sp->m_buckets_)), sizeof(spPage *) * number_of_cell);

    spParallelMemset((void *) ((sp->m_buckets_)), 0x0, sizeof(spPage *) * number_of_cell);


//    LOAD_KERNEL(spParticleDeployKernel,
//                sizeType2Dim3(spMeshGetShape(sp->m)), NUMBER_OF_THREADS_PER_BLOCK,
//                sp->m_buckets_,
//                sp->m_pages_,
//                sp->max_number_of_pages);

    spParallelDeviceSync();        //wait for iteration to finish

    DONE
}

size_type **spParticleBuckets(spParticle *sp) { return sp->m_buckets_; };

size_type **spParticlePagePool(spParticle *sp) { return sp->m_page_pool_; };

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

void spParticleSyncStart(spParticle *sp)
{
    int num_of_neighbour = spMPITopologyNumOfNeighbours();

//    MPI_Neighbor_allgather(&rank, 1, MPI_INT, buffer, 1, MPI_INT, spMPIComm());

    MPI_Comm comm = spMPIComm();

    {
        int topo_type;
        MPI_Topo_test(comm, &topo_type);
        assert(topo_type == MPI_CART);
    }

    int mpi_topology_ndims = 0;


    MPI_Cartdim_get(comm, &mpi_topology_ndims);

    for (int i = 0; i < mpi_topology_ndims; ++i)
    {

        int r0, r1;
        MPI_Cart_shift(spMPIComm(), 0, 1, &r0, &r1);

        int request_count = 0;
        MPI_Request requests[sp->num_of_attrs * 4];
        int sizes[mpi_topology_ndims];

        MPI_Aint send_displaces[mpi_topology_ndims * 2];
        MPI_Aint recv_displaces[mpi_topology_ndims * 2];


        for (int j = 0; j < sp->num_of_attrs; ++j)
        {

            MPI_Isend(send_displaces[i * 2],
                      1,
                      x_dir_type,
                      r0,
                      j,
                      spMPIComm(), &requests[request_count]);
            ++request_count;

            MPI_Irecv(recv_displaces[i * 2],
                      1,
                      x_dir_type,
                      r0,
                      j,
                      spMPIComm(), &requests[request_count]);

            ++request_count;

            MPI_Isend(send_displaces[i * 2 + 1],
                      1,
                      x_dir_type,
                      r1,
                      j,
                      spMPIComm(), &requests[request_count]);

            ++request_count;
            MPI_Irecv(recv_displaces[i * 2 + 1],
                      1,
                      x_dir_type,
                      r1,
                      j,
                      spMPIComm(), &requests[request_count]);
            ++request_count;
        }

        MPI_Waitall(request_count, requests, MPI_STATUS_IGNORE);

    }

//    free(buffer);

//    int num_of_pages_send[MAX_NUM_OF_NEIGHBOUR];
//    int num_of_pages_recv[MAX_NUM_OF_NEIGHBOUR];
//
//    int *page_offset_send[MAX_NUM_OF_NEIGHBOUR];
//    int *page_offset_recv[MAX_NUM_OF_NEIGHBOUR];
//
//    // sync number of pages
//    int num_reqs = 0;
//
//    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
//    {
//
//        page_offset_send[i] = NULL;
//
//        num_of_pages_recv[i] = 0;
//
//        num_of_pages_send[i] = 0;
//
//
//        size_type lower[3], upper[3];
//
//        int offset[3];
//
//        if (spMeshGetDomain(sp->m, i, lower, upper, offset) == 0) { continue; }
////        if (spMPITopologyNeighbours(offset) == spMPIRank()) { continue; }
//
//        int dest = 0, send_tag = 0, recv_tag;
//
//        spMPIMakeSendRecvTag(sp->id, offset, &dest, &send_tag, &recv_tag);
//
////        num_of_pages_send[i] = (int) spParticlePageExpand(sp, lower, upper, 1024, &(page_offset_send[i]));
//
//
//        MPI_ERROR(MPI_Isend(&(num_of_pages_send[i]),
//                            1,
//                            MPI_INT,
//                            dest,
//                            send_tag,
//                            spMPIComm(),
//                            &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
//        ++(num_reqs);
//
//        MPI_ERROR(MPI_Irecv(&(num_of_pages_recv[i]),
//                            1,
//                            MPI_INT,
//                            dest,
//                            recv_tag,
//                            spMPIComm(),
//                            &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
//        ++(num_reqs);
//
//    }
//
//    MPI_ERROR(MPI_Waitall(sp->sync_reqs.num_reqs, sp->sync_reqs.requests, MPI_STATUS_IGNORE));

//
//    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
//    {
//        if (num_of_pages_recv[i] <= 1) { continue; }
//
//        spParallelHostAlloc((void **) &(page_offset_recv[i]), sizeof(int) * num_of_pages_recv[i]);
//
//    }
//
//    sp->sync_reqs.num_reqs = 0;
//
//    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
//    {
//
//        size_type lower[3], upper[3];
//        int offset[3];
//
//        if (spMeshGetDomain(sp->m, i, lower, upper, offset) == 0) { continue; }
////        if (spMPITopologyNeighbours(offset) == spMPIRank()) { continue; }
//        int dest = 0, send_tag = 0, recv_tag;
//
//        spMPIMakeSendRecvTag(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR, offset, &dest, &send_tag, &recv_tag);
//
//
//        for (int s = 0; s < sp->num_of_attrs; ++s)
//        {
//            int tag = (int) (sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR) + s;
//            spMPIMakeSendRecvTag(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR + s + 1, offset, &dest, &tag, &tag);
//
//            MPI_Datatype send_datatype;
//
//            spMPIDataTypeCreate(num_of_pages_send[i],
//                                page_offset_send[i],
//                                sp->attrs[s].type_tag,
//                                &send_datatype);
//
//
//            MPI_ERROR(MPI_Isend(
//                sp->attrs[s].data,
//                1,
//                send_datatype,
//                dest,
//                tag,
//                spMPIComm(),
//                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
//
//            MPI_ERROR(MPI_Type_free(&send_datatype));
//
//            ++sp->sync_reqs.num_reqs;
//
//
//            MPI_Datatype recv_datatype;
//
////            spMPIDataTypeCreate(num_of_pages_recv,
////                                page_offset_recv,
////                                sp->attrs[s].type_tag,
////                                &recv_datatype);
//
//            MPI_ERROR(MPI_Irecv(
//                sp->attrs[s].data,
//                1,
//                recv_datatype,
//                MPI_ANY_SOURCE,
//                tag,
//                spMPIComm(),
//                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
//
//            ++sp->sync_reqs.num_reqs;
//
//            MPI_ERROR(MPI_Type_free(&recv_datatype));
//
//        }
//    }
}

void spParticleSyncEnd(spParticle *sp)
{
//    MPI_Waitall(sp->sync_reqs.num_reqs, sp->sync_reqs.requests, MPI_STATUS_IGNORE);

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

//    char curr_path[2048];
//    char new_path[2048];
//    strcpy(new_path, name);
//    new_path[strlen(name)] = '/';
//    new_path[strlen(name) + 1] = '\0';
//
//
//    spIOStreamPWD(os, curr_path);
//    spIOStreamOpen(os, new_path);
//
//
//    int *page_offset_host = NULL;
//
//    size_type lower[3], upper[3];
//
//    spMeshGetDomain(sp->m, SP_DOMAIN_CENTER, lower, upper, NULL);
//
//    int num_of_pages = 0;// spParticlePageExpand(sp, lower, upper, sp->max_number_of_pages, &page_offset_host);
//
//    if (num_of_pages > 0)
//    {
//        size_type num_of_entities = (size_type) (SP_NUMBER_OF_ENTITIES_IN_PAGE * num_of_pages);
//
//        int count[2] = {num_of_pages, SP_NUMBER_OF_ENTITIES_IN_PAGE};
//        int topology_dims[2] = {0, SP_NUMBER_OF_ENTITIES_IN_PAGE};
//        int start[2] = {0, 0};
//
//        MPI_Scan(&num_of_pages, &topology_dims[0], 1, MPI_INT, MPI_SUM, spMPIComm());
//        start[0] = topology_dims[0] - num_of_pages;
//        {
//
//            MeshEntityId *page_id_host = NULL;
//
//            spParallelHostAlloc((void **) &page_id_host, num_of_pages * sizeof(MeshEntityId));
//
////            spParallelMemcpyIndexedBlock(page_id_host,
////                                         sp->m_ids_,
////                                         (int) num_of_pages,
////                                         sizeof(MeshEntityId),
////                                         page_offset_host);
//
//
//            spIOStreamWriteSimple(os,
//                                  "id",
//                                  SP_TYPE_int64_t,
//                                  page_id_host,
//                                  1,
//                                  topology_dims,
//                                  start,
//                                  NULL,
//                                  count,
//                                  NULL,
//                                  SP_FILE_APPEND);
//
//            spParallelHostFree((void **) &page_id_host);
//        }
//
//        for (int i = 0, ie = sp->num_of_attrs; i < ie; ++i)
//        {
//            void *write_buffer;
//
//            size_type page_size_in_byte = sp->attrs[i].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;
//
//            spParallelHostAlloc(&write_buffer, num_of_entities * sp->attrs[i].size_in_byte);
//
//            for (int j = 0; j < num_of_pages; ++j)
//            {
//                spParallelMemcpy((byte_type *) (write_buffer) + j * page_size_in_byte,
//                                 (byte_type *) (sp->attrs[i].data) + j * page_size_in_byte,
//                                 page_size_in_byte);
//            }
//
//
//            spIOStreamWriteSimple(os,
//                                  sp->attrs[i].name,
//                                  sp->attrs[i].type_tag,
//                                  write_buffer,
//                                  2,
//                                  count,
//                                  NULL,
//                                  NULL,
//                                  NULL,
//                                  NULL,
//                                  SP_FILE_APPEND);
//
//            spParallelHostFree(&write_buffer);
//
//        }
//    }
//    spParallelHostFree((void **) &page_offset_host);
//
//    spIOStreamOpen(os, curr_path);

}
void spParticleRead(struct spParticle_s *f, spIOStream *os, char const url[], int flag)
{

}

