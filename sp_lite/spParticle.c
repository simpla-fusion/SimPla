//
// Created by salmon on 16-7-20.
//
#include "sp_lite_def.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>


#include "spParallel.h"
#include "spMesh.h"
#include "spObject.h"
#include "spParticle.h"


struct spParticleAttrEntity_s
{
    int type_tag;
    size_type size_in_byte;
    size_type aligned_size;
    size_type offset;

    char name[255];
    size_type page_offset;
    void *data;
};

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

    size_type m_page_size_in_byte_;

    size_type m_entity_size_in_byte_;

    spParticlePage *m_data_root_; // DEVICE

    spParticlePage **m_page_pool_;// DEVICE

    spParticlePage **m_base_;  // DEVICE  base manifold B

    size_type *m_page_count_;// DEVICE

};

int spParticleCreate(spParticle **sp, const spMesh *mesh)
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

    spParallelDeviceAlloc((void **) (&(sp->m_base_)), sizeof(spParticlePage *) * number_of_cell);

    spParallelMemset((void *) ((sp->m_base_)), 0x0, sizeof(spParticlePage *) * number_of_cell);


    sp->m_num_of_entities_in_page_ = SP_NUMBER_OF_ENTITIES_IN_PAGE;

    sp->m_max_num_of_pages_ = number_of_cell * (PIC * 2 / sp->m_num_of_entities_in_page_ + 1);

    sp->m_page_size_in_byte_ = 0;

    sp->m_entity_size_in_byte_ = 0;

    for (int i = 0; i < sp->m_num_of_attrs_; ++i)
    {

        sp->m_page_size_in_byte_ += sp->m_attrs_->aligned_size * sp->m_num_of_entities_in_page_;

        sp->m_entity_size_in_byte_ += sp->m_attrs_->size_in_byte;

    }
    sp->m_page_size_in_byte_ *= sp->m_num_of_entities_in_page_;

    spParallelDeviceAlloc((void **) &(sp->m_data_root_), sp->m_page_size_in_byte_ * sp->m_max_num_of_pages_);

    spParallelDeviceAlloc((void **) (&(sp->m_page_pool_)), sizeof(spParticlePage *));

    spParallelMemcpy((void *) (sp->m_page_pool_), sp->m_data_root_, sizeof(spParticlePage *));

    spParallelDeviceAlloc((void **) (&(sp->m_page_count_)), sizeof(int) * number_of_cell);

    spParallelMemset((void *) ((sp->m_page_count_)),
                     (PIC / sp->m_num_of_entities_in_page_ + 1),
                     sizeof(spParticlePage *) * number_of_cell);

    spParticleResizePageLink(sp);

    return SP_SUCCESS;

}

int spParticleDestroy(spParticle **sp)
{

    spParallelDeviceFree((void **) &((*sp)->m_data_root_));
    spParallelDeviceFree((void **) &((*sp)->m_base_));
    spParallelDeviceFree((void **) &((*sp)->m_page_pool_));

    spParallelHostFree(sp);

    return SP_SUCCESS;
}

spParticlePage *spParticleDataRoot(spParticle *sp) { return sp->m_data_root_; };

spParticlePage **spParticleBaseField(spParticle *sp) { return sp->m_base_; };

spParticlePage **spParticlePagePool(spParticle *sp) { return sp->m_page_pool_; };

size_type *spParticlePageCount(spParticle *sp) { return sp->m_page_count_; };

size_type spParticleNumOfEntitiesInPage(spParticle const *sp) { return sp->m_num_of_entities_in_page_; };

spMesh const *spParticleMesh(spParticle const *sp) { return sp->m; };

Real spParticleMass(spParticle const *sp) { return sp->mass; }

Real spParticleCharge(spParticle const *sp) { return sp->charge; }

int spParticleAddAttribute(struct spParticle_s *pg,
                           char const *name,
                           int type_tag,
                           size_type size_in_byte,
                           size_type page_offset)
{
    strcpy(pg->m_attrs_[pg->m_num_of_attrs_].name, name);

    pg->m_attrs_[pg->m_num_of_attrs_].type_tag = type_tag;

    pg->m_attrs_[pg->m_num_of_attrs_].size_in_byte = size_in_byte;

    pg->m_attrs_[pg->m_num_of_attrs_].page_offset = page_offset;

    if (pg->m_num_of_attrs_ == 0)
    {
        pg->m_attrs_[pg->m_num_of_attrs_].offset = 0;
    }
    else
    {
        pg->m_attrs_[pg->m_num_of_attrs_].offset = (pg->m_attrs_[pg->m_num_of_attrs_ - 1].offset
            + pg->m_attrs_[pg->m_num_of_attrs_ - 1].size_in_byte);
    }


    ++pg->m_num_of_attrs_;

    assert(pg->m_num_of_attrs_ < SP_MAX_NUMBER_OF_PARTICLE_ATTR);
}

void *spParticleAttributeData(struct spParticle_s *pg, int i)
{
    return i < pg->m_num_of_attrs_ ? pg->m_attrs_[i].data : NULL;
};

void **spParticleAttributeDeviceData(struct spParticle_s *pg)
{
    return (void **) (pg->m_data_root_);
};

int spParticleGetibuteTypeTag(struct spParticle_s *pg, int i)
{
    return i < pg->m_num_of_attrs_ ? pg->m_attrs_[i].type_tag : 0;
};

size_type spParticleAttibuteSizeInByte(struct spParticle_s *pg, int i)
{
    return i < pg->m_num_of_attrs_ ? pg->m_attrs_[i].size_in_byte : 0;
}

void spParticleAttributeName(struct spParticle_s *pg, int i, char *name)
{
    if (i < pg->m_num_of_attrs_) { strcpy(name, pg->m_attrs_[i].name); }
}

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

    {
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
                                    &send_disps[2 * d + 0]);
            spParticleGetPageOffset(sp,
                                    s_start_lower,
                                    s_count_lower,
                                    &send_block_count[2 * d + 1],
                                    &send_disps[2 * d + 1]);
            spParticleGetPageOffset(sp,
                                    r_start_lower,
                                    r_count_lower,
                                    &recv_block_count[2 * d + 0],
                                    &recv_disps[2 * d + 0]);
            spParticleGetPageOffset(sp,
                                    r_start_lower,
                                    r_count_lower,
                                    &recv_block_count[2 * d + 1],
                                    &recv_disps[2 * d + 1]);
        }
    }
    size_type block_length = sp->m_num_of_entities_in_page_;

    MPI_Comm comm = spMPIComm();

    MPI_Datatype ele_type;

//        spMPIDataTypeCreate(sp->m_attrs_[i].type_tag,(sp->m_attrs_[i].size_in_byte), &ele_type);

    MPI_Type_contiguous((int) (sp->m_page_size_in_byte_), MPI_BYTE, &ele_type);

    spUpdateIndexedBlock(sp->m_data_root_,
                         (size_type const **) send_disps,
                         send_block_count,
                         sp->m_data_root_,
                         (size_type const **) recv_disps,
                         recv_block_count,
                         block_length,
                         ele_type,
                         comm);

    MPI_Type_free(&ele_type);

//    MPI_Comm comm = spMPIComm();
//
//    {
//        int topo_type;
//        MPI_Topo_test(comm, &topo_type);
//        assert(topo_type == MPI_CART);
//    }
//
//    int mpi_topology_ndims = 0;
//
//
//    MPI_Cartdim_get(comm, &mpi_topology_ndims);
//
//    for (int i = 0; i < mpi_topology_ndims; ++i)
//    {
//
//        int r0, r1;
//        MPI_Cart_shift(spMPIComm(), 0, 1, &r0, &r1);
//
//        int request_count = 0;
//        MPI_Request requests[sp->m_num_of_attrs_ * 4];
//        int sizes[mpi_topology_ndims];
//
//        MPI_Aint send_displaces[mpi_topology_ndims * 2];
//        MPI_Aint recv_displaces[mpi_topology_ndims * 2];
//
//
//        for (int j = 0; j < sp->m_num_of_attrs_; ++j)
//        {
//
//            MPI_Isend(send_displaces[i * 2],
//                      1,
//                      x_dir_type,
//                      r0,
//                      j,
//                      spMPIComm(), &requests[request_count]);
//            ++request_count;
//
//            MPI_Irecv(recv_displaces[i * 2],
//                      1,
//                      x_dir_type,
//                      r0,
//                      j,
//                      spMPIComm(), &requests[request_count]);
//
//            ++request_count;
//
//            MPI_Isend(send_displaces[i * 2 + 1],
//                      1,
//                      x_dir_type,
//                      r1,
//                      j,
//                      spMPIComm(), &requests[request_count]);
//
//            ++request_count;
//            MPI_Irecv(recv_displaces[i * 2 + 1],
//                      1,
//                      x_dir_type,
//                      r1,
//                      j,
//                      spMPIComm(), &requests[request_count]);
//            ++request_count;
//        }
//
//        MPI_Waitall(request_count, requests, MPI_STATUS_IGNORE);
//
//    }
//
//    free(send_buffer);
//
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
//        if (spMeshDomain(sp->m, i, lower, upper, offset) == 0) { continue; }
//       if (spMPITopologyNeighbours(offset) == spMPIRank()) { continue; }
//
//        int dest = 0, send_tag = 0, recv_tag;
//
//        spMPIMakeSendRecvTag(sp->id, offset, &dest, &send_tag, &recv_tag);
//
//        num_of_pages_send[i] =spParticlePageExpand(sp, lower, upper, 1024, &(page_offset_send[i]));
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
//        if (spMeshDomain(sp->m, i, lower, upper, offset) == 0) { continue; }
////        if (spMPITopologyNeighbours(offset) == spMPIRank()) { continue; }
//        int dest = 0, send_tag = 0, recv_tag;
//
//        spMPIMakeSendRecvTag(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR, offset, &dest, &send_tag, &recv_tag);
//
//
//        for (int s = 0; s < sp->m_num_of_attrs_; ++s)
//        {
//            int tag =(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR) + s;
//            spMPIMakeSendRecvTag(sp->id * SP_MAX_NUMBER_OF_PARTICLE_ATTR + s + 1, offset, &dest, &tag, &tag);
//
//            MPI_Datatype send_datatype;
//
//            spMPIDataTypeCreate(num_of_pages_send[i],
//                                page_offset_send[i],
//                                sp->m_attrs_[s].type_tag,
//                                &send_datatype);
//
//
//            MPI_ERROR(MPI_Isend(
//                sp->m_attrs_[s].data,
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
//                sp->m_attrs_[s].data,
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
//    spMeshDomain(sp->m, SP_DOMAIN_CENTER, lower, upper, NULL);
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
////                                        num_of_pages,
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
//        for (int i = 0, ie = sp->m_num_of_attrs_; i < ie; ++i)
//        {
//            void *write_buffer;
//
//            size_type page_size_in_byte = sp->m_attrs_[i].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;
//
//            spParallelHostAlloc(&write_buffer, num_of_entities * sp->m_attrs_[i].size_in_byte);
//
//            for (int j = 0; j < num_of_pages; ++j)
//            {
//                spParallelMemcpy((byte_type *) (write_buffer) + j * page_size_in_byte,
//                                 (byte_type *) (sp->m_attrs_[i].data) + j * page_size_in_byte,
//                                 page_size_in_byte);
//            }
//
//
//            spIOStreamWriteSimple(os,
//                                  sp->m_attrs_[i].name,
//                                  sp->m_attrs_[i].type_tag,
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

