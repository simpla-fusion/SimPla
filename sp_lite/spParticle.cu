/**
 * @file spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "sp_lite_def.h"
#include "spParallel.h"

#include "spMesh.h"
#include "spPage.h"
#include "spParticle.h"


typedef struct spParticleData_s
{
    void *attrs[SP_MAX_NUMBER_OF_PARTICLE_ATTR];
} spParticleData;

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

    int num_of_pages_send[MAX_NUM_OF_NEIGHBOUR];
    int num_of_pages_recv[MAX_NUM_OF_NEIGHBOUR];

    MeshEntityId *page_id_send_buffer[MAX_NUM_OF_NEIGHBOUR];
    MeshEntityId *page_id_recv_buffer[MAX_NUM_OF_NEIGHBOUR];

    int *page_offset_send[MAX_NUM_OF_NEIGHBOUR];
    int *page_offset_recv[MAX_NUM_OF_NEIGHBOUR];

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

    size_type number_of_pages;
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

MC_GLOBAL void spParticleTestAtomicPageOp(spParticlePage **pg)
{
    MC_SHARED spPage *p;

    spParallelSyncThreads();

    if (spParallelThreadNum() == 0) { p = (spPage *) (*pg); }

    spParallelSyncThreads();

    spPage *res = spPageAtomicPop(&p);

    if (spParallelThreadNum() == 0) { (*pg) = (spParticlePage *) p; }

}
MC_GLOBAL void spParticleDeployKernel(spParticlePage **bucket, spParticlePage *pg, size_type num_of_pages)
{
    size_type offset = spParallelBlockNum() * spParallelNumOfThreads() + spParallelThreadNum();
    for (size_type pos = offset; pos < offset + spParallelNumOfThreads() && pos < num_of_pages;
         pos += spParallelNumOfThreads())
    {
        pg[pos].next = &(pg[pos + 1]);
        pg[pos].id.v = 0;
        pg[pos].offset = pos * SP_NUMBER_OF_ENTITIES_IN_PAGE;
    }

    if (spParallelThreadNum() == 0) { bucket[spParallelBlockNum()] = NULL; }
}
void spParticleDeploy(spParticle *sp, size_type PIC)
{
    size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, sp->iform/*volume*/);

    size_type num_page_per_cell = PIC * 2 / SP_NUMBER_OF_ENTITIES_IN_PAGE + 1;

    sp->number_of_pages = number_of_cell * num_page_per_cell;

    void *t_data[SP_MAX_NUMBER_OF_PARTICLE_ATTR];

    sp->max_num_of_entities = sp->number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;
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

    spParallelDeviceMalloc((void **) (&(sp->m_pages_)), sp->number_of_pages * sizeof(spParticlePage));

    spParallelDeviceMalloc((void **) (&(sp->m_page_pool_)), sizeof(spPage *));

    spParallelMemcpy((void *) (sp->m_page_pool_), (void const *) &(sp->m_pages_), sizeof(spPage *));

    spParallelDeviceMalloc((void **) (&(sp->m_buckets_)), sizeof(spPage *) * number_of_cell);

    spParallelMemset((void *) ((sp->m_buckets_)), 0x0, sizeof(spPage *) * number_of_cell);


    LOAD_KERNEL(spParticleDeployKernel,
                sizeType2Dim3(spMeshGetShape(sp->m)), NUMBER_OF_THREADS_PER_BLOCK,
                sp->m_buckets_,
                sp->m_pages_,
                sp->number_of_pages);

    spParallelDeviceSync();        //wait for iteration to finish

    DONE
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
    if (offset == int(-1))
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

enum { SP_COPY_OUT_ONLY_DATA = 1, SP_COPY_OUT_WHOLE_PAGE = 2 };

//MC_GLOBAL void spParticleCountKernel(dim3 dim, dim3 offset, dim3 count, spParticlePage **buckets,
//                                     int *page_count_device)
//{
//    dim3 idx = spParallelBlockIdx();
//
//    if (count.x > idx.x && count.y > idx.y && count.z > idx.z)
//    {
//        int pos = idx.x + offset.x + (idx.y + offset.y + (idx.z + offset.z) * dim.y) * dim.x;
//        int pos2 = idx.x + (idx.y + idx.z * count.y) * count.x;
//
//        int pg_count = 0;
//
//        spParticlePage *pg = buckets[pos];
//
//        while (pg != NULL)
//        {
//            ++pg_count;
//            pg = pg->next;
//        }
//
//        page_count_device[pos2] = pg_count;
//    }
//}


MC_GLOBAL void spParticlePageExpandKernel(dim3 dims, dim3 lower,
                                          spParticlePage **buckets,
                                          MeshEntityId *out_id,
                                          int *out_offset,
                                          int *pos)
{
    if (spParallelThreadNum() == 0)
    {
        dim3 idx = spParallelBlockIdx();

        idx.x += lower.x;
        idx.y += lower.y;
        idx.z += lower.z;

        spParticlePage *pg = buckets[idx.x + (idx.y + idx.z * dims.y) * dims.x];

        while (pg != NULL)
        {
            int s = spAtomicAdd(pos, 1);
            out_id[s] = pg->id;
            out_offset[s] = (int) (pg->offset);
            pg = pg->next;
        }
    }
}
int spParticlePageExpand(spParticle const *sp,
                         size_type const *lower,
                         size_type const *upper,
                         size_type max_num_of_pages,
                         MeshEntityId **out_id_host,
                         int **out_offset_host)
{

    size_type count[3];
    for (int i = 0; i < 3; ++i)
    {
        count[i] = (upper[i] - lower[i]);
    }


    size_type num_of_cell = count[0] * count[1] * count[2];

    if (num_of_cell == 0) { return 0; }

    max_num_of_pages = num_of_cell * 2;

    int num_of_page = 0;

    MeshEntityId *out_id_device;

    int *out_offset_device = NULL;

    int *num_of_page_device = NULL;

    spParallelDeviceMalloc((void **) (&num_of_page_device), sizeof(int));

    spParallelMemset((void *) (num_of_page_device), 0, sizeof(int));

    spParallelDeviceMalloc((void **) (&out_id_device), max_num_of_pages * sizeof(MeshEntityId));

    spParallelDeviceMalloc((void **) (&out_offset_device), max_num_of_pages * sizeof(int));

    LOAD_KERNEL(spParticlePageExpandKernel,
                sizeType2Dim3(count), 1,
                sizeType2Dim3(spMeshGetShape(sp->m)),
                sizeType2Dim3(lower),
                sp->m_buckets_,
                out_id_device,
                out_offset_device,
                num_of_page_device);

    spParallelMemcpy((void *) (&num_of_page), (void *) (num_of_page_device), sizeof(int));

    spParallelDeviceFree((void **) (&num_of_page_device));


    if (num_of_page > 0)
    {

        spParallelHostMalloc((void **) (out_id_host), sizeof(MeshEntityId) * num_of_page);

        spParallelMemcpy((void *) (*out_id_host), (void *) (out_id_device), num_of_page * sizeof(MeshEntityId));

        spParallelHostMalloc((void **) (out_offset_host), sizeof(int) * num_of_page);

        spParallelMemcpy((void *) (*out_offset_host), (void *) (out_offset_device), num_of_page * sizeof(int));

    }

    spParallelDeviceFree((void **) (&out_id_device));

    spParallelDeviceFree((void **) (&out_offset_device));

    return num_of_page;
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


    MeshEntityId *page_id_host = NULL;
    int *page_offset_host = NULL;

    size_type lower[3], upper[3];
    spMeshGetDomain(sp->m, 0, lower, upper, NULL);
    int num_of_pages =
        spParticlePageExpand(
            sp, lower, upper,
            sp->number_of_pages,
            &page_id_host,
            &page_offset_host);


    if (num_of_pages > 0)
    {
        size_type num_of_entities = (size_type) (SP_NUMBER_OF_ENTITIES_IN_PAGE * num_of_pages);


        for (int i = 0, ie = sp->num_of_attrs; i < ie; ++i)
        {
            void *write_buffer;

            size_type page_size_in_byte = sp->attrs[i].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;

            spParallelHostMalloc(&write_buffer, num_of_entities * sp->attrs[i].size_in_byte);

            for (int j = 0; j < num_of_pages; ++j)
            {
                spParallelMemcpy(write_buffer + j * page_size_in_byte,
                                 sp->attrs[i].data + j * page_size_in_byte,
                                 page_size_in_byte);
            }


            size_type start = 0;

            spIOStreamWriteSimple(os,
                                  sp->attrs[i].name,
                                  sp->attrs[i].type_tag,
                                  write_buffer,
                                  1,
                                  &num_of_entities,
                                  &start,
                                  NULL,
                                  &num_of_entities,
                                  NULL,
                                  SP_FILE_APPEND);

            spParallelHostFree(&write_buffer);

        }
    }
    spParallelHostFree((void **) &page_id_host);
    spParallelHostFree((void **) &page_offset_host);

    spIOStreamOpen(os, curr_path);

}

void spParticleRead(spParticle *f, char const url[], int flag)
{

}

/*****************************************************************************************/

MC_GLOBAL void spParticlePopPageKernel(spParticlePage **pool, int num, spParticlePage **page_ptr_device)
{


    if (spParallelBlockNum() == 0 && spParallelThreadNum() == 0)
    {
        for (int i = 0; i < num; ++i)
        {
            page_ptr_device[i] = (spParticlePage *) spPageAtomicPop((spPage **) pool);
        }
    }
}
void spParticlePopPages(spParticle const *sp, int num, spParticlePage **page_ptr_host, int *pos)
{
    spParticlePage **page_data_ptr_device;


    spParallelDeviceMalloc((void **) (&page_data_ptr_device), num * sizeof(spPage *));

    LOAD_KERNEL(spParticlePopPageKernel, 1, 1, sp->m_page_pool_, num, page_data_ptr_device);

    spParallelHostMalloc((void **) (page_ptr_host), num * sizeof(spPage *));

    spParallelMemcpy((void *) (*page_ptr_host),
                     (void *) (page_data_ptr_device),
                     num * sizeof(spPage *));

    spParallelDeviceFree((void **) &page_data_ptr_device);
}

void spParticleCreateMPIDataType(spParticle *sp,
                                 int type_tag,
                                 int count,
                                 int *array_of_displacements,
                                 MPI_Datatype *new_type
)
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
    MPI_Datatype old_type;

//    MPI_ERROR(MPI_Type_contiguous((int) sp->max_num_of_entities, ele_type, &old_type));
//    MPI_ERROR(MPI_Type_commit(&old_type));
    int block = SP_NUMBER_OF_ENTITIES_IN_PAGE;
    MPI_ERROR(MPI_Type_create_indexed_block(count, block, array_of_displacements, ele_type, new_type));
    MPI_ERROR(MPI_Type_commit(new_type));
//    MPI_ERROR(MPI_Type_free(&old_type));
}
void spParticleSyncStart(spParticle *sp)
{

    // sync number of pages
    sp->sync_reqs.num_reqs = 0;

    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
    {

        size_type lower[3], upper[3];

        int offset[3];

        if (spMeshGetDomain(sp->m, i, lower, upper, offset) == 0) { continue; }

        int dest = 0, send_tag = 0, recv_tag;

        spMPIMakeSendRecvTag(sp->id, offset, &dest, &send_tag, &recv_tag);

        sp->sync_reqs.page_id_send_buffer[i] = NULL;

        sp->sync_reqs.page_offset_send[i] = NULL;

        sp->sync_reqs.num_of_pages_recv[i] = 0;

        sp->sync_reqs.num_of_pages_send[i] =
            spParticlePageExpand(sp, lower, upper,
                                 1024,
                                 &(sp->sync_reqs.page_id_send_buffer[i]),
                                 &(sp->sync_reqs.page_offset_send[i])
            );


        if (dest > 0)
        {
            MPI_ERROR(MPI_Isend(&(sp->sync_reqs.num_of_pages_send[i]),
                                1,
                                MPI_INT,
                                dest,
                                send_tag,
                                spMPIComm(),
                                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
            ++(sp->sync_reqs.num_reqs);

            MPI_ERROR(MPI_Irecv(&(sp->sync_reqs.num_of_pages_recv[i]),
                                1,
                                MPI_INT,
                                dest,
                                recv_tag,
                                spMPIComm(),
                                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));
            ++(sp->sync_reqs.num_reqs);
        }
    }

    MPI_ERROR(MPI_Waitall(sp->sync_reqs.num_reqs, sp->sync_reqs.requests, MPI_STATUS_IGNORE));


    sp->sync_reqs.num_reqs = 0;
    CHECK_INT(sp->num_of_attrs)
    for (int i = 0; i < MAX_NUM_OF_NEIGHBOUR; ++i)
    {

        size_type lower[3], upper[3];
        int offset[3];

        if (spMeshGetDomain(sp->m, i, lower, upper, offset) == 0) { continue; }

        for (int s = 0; s < sp->num_of_attrs; ++s)
        {
            int dest = 0, send_tag = 0, recv_tag;

            spMPIMakeSendRecvTag(sp->id, offset, &dest, &send_tag, &recv_tag);

            MPI_Datatype send_datatype;

            spParticleCreateMPIDataType(
                sp,
                sp->attrs[s].type_tag,
                sp->sync_reqs.num_of_pages_send[i],
                sp->sync_reqs.page_offset_send[i],
                &send_datatype);


            MPI_ERROR(MPI_Isend(
                sp->attrs[s].data,
                1,
                send_datatype,
                dest,
                send_tag,
                spMPIComm(),
                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));

            MPI_ERROR(MPI_Type_free(&send_datatype));

            ++sp->sync_reqs.num_reqs;

            MPI_Datatype recv_datatype;

            spParticleCreateMPIDataType(
                sp,
                sp->attrs[s].type_tag,
                sp->sync_reqs.num_of_pages_recv[i],
                sp->sync_reqs.page_offset_recv[i],
                &recv_datatype);

            MPI_ERROR(MPI_Irecv(
                sp->attrs[s].data,
                1,
                recv_datatype,
                dest,
                recv_tag,
                spMPIComm(),
                &(sp->sync_reqs.requests[sp->sync_reqs.num_reqs])));

            ++sp->sync_reqs.num_reqs;

            MPI_ERROR(MPI_Type_free(&recv_datatype));

        }
    }
    CHECK_INT(sp->sync_reqs.num_reqs);
}

void spParticleSyncEnd(spParticle *sp)
{
    MPI_Waitall(sp->sync_reqs.num_reqs, sp->sync_reqs.requests, MPI_STATUS_IGNORE);

    for (int j = 0; j < MAX_NUM_OF_NEIGHBOUR; ++j)
    {
        for (int i = 0; i < sp->sync_reqs.num_of_pages_send[j]; ++i)
        {
            spParallelHostFree((void **) &(sp->sync_reqs.page_offset_send[i]));
        }
        for (int i = 0; i < sp->sync_reqs.num_of_pages_recv[j]; ++i)
        {
            spParallelHostFree((void **) &(sp->sync_reqs.page_offset_recv[i]));
        }

    }
}

void spParticleSync(spParticle *sp)
{
    spParticleSyncStart(sp);

    spParticleSyncEnd(sp);
}
//
//MC_DEVICE int spPageInsert(spPage **dest, spPage **pool, int *d_tail, int *g_d_tail)
//{
//	while (1)
//	{
//		if ((*dest) != NULL)
//		{
//			while ((*d_tail = spAtomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//			{
//				if ((P_GET_FLAG((*dest)->data, *d_tail).v == 0))
//				{
//					break;
//				}
//			}
//		}
//
//		spParallelSyncThreads();
//		if ((*dest) == NULL)
//		{
//
//			if (spParallelThreadNum() == 0)
//			{
//				*dest = *pool;
//				*pool = (*pool)->next;
//				if (*dest != NULL)
//				{
//					(*dest)->next = NULL;
//				}
//				*g_d_tail = 0;
//			}
//
//		}
//		else if (*d_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
//		{
//			dest = &((*dest)->next);
//			if (*dest != NULL)
//			{
//				(*g_d_tail) = 0;
//			}
//
//		}
//		spParallelSyncThreads();
//
//		if (*dest == NULL)
//		{
//			return SP_MP_ERROR_POOL_IS_OVERFLOW;
//		}
//	}
//
//	return SP_MP_FINISHED;
//}
//
//MC_DEVICE int spPageScan(spPage **dest, int *d_tail, int *g_d_tail, int MASK, int tag)
//{
//	int THREAD_ID = spParallelThreadNum();
//
//	while ((*dest) != NULL)
//	{
//		while ((*d_tail = spAtomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//		{
//			if ((P_GET_FLAG((*dest)->data, *d_tail).v & MASK == tag & MASK))
//			{
//				return SP_MP_SUCCESS;
//			}
//		}
//
//		spParallelSyncThreads();
//
//		dest = &((*dest)->next);
//
//		if (THREAD_ID == 0)
//		{
//			g_d_tail = 0;
//		}
//
//		spParallelSyncThreads();
//
//	}
//	return SP_MP_FINISHED;
//}
//
//#define SP_MAP(_P_DEST_, _POS_DEST_, _P_SRC_, _POS_SRC_, _ENTITY_SIZE_IN_BYTE_)   spParallelMemcpy(_P_DEST_ + _POS_DEST_ * _ENTITY_SIZE_IN_BYTE_,_P_SRC_ + _POS_SRC_ * _ENTITY_SIZE_IN_BYTE_,   _ENTITY_SIZE_IN_BYTE_);
//
//MC_DEVICE void spUpdateParticleSortThreadKernel(spPage **dest, spPage const **src, spPage **pool,
//		int entity_size_in_byte, int MASK, MeshEntityId tag)
//{
//
//	MC_SHARED
//	int g_d_tail, g_s_tail;
//
//	spParallelSyncThreads();
//
//	if (spParallelThreadNum() == 0)
//	{
//		g_s_tail = 0;
//		g_d_tail = 0;
//	}
//	spParallelSyncThreads();
//
//	for (int d_tail = 0, s_tail = 0;
//			spPageMapAndPack(dest, src, &d_tail, &g_d_tail, &s_tail, &g_s_tail, /* pool, MASK,*/tag) != SP_MP_FINISHED;)
//	{
////		SP_MAP(byte_type*)((*dest)->data), d_tail, (byte_type*) ((*src)->data), s_tail, entity_size_in_byte);
//	}
//
//}

//MC_DEVICE int spParticleMapAndPack(void **data,
//                                   spParticlePage **dest,
//                                   spParticlePage **src,
//                                   int *d_tail,
//                                   int *g_d_tail,
//                                   int *s_tail,
//                                   int *g_s_tail,
//                                   spParticlePage **pool,
//                                   MeshEntityId tag)
//{
//
//    while (*src != NULL)
//    {
//        if ((*dest) != NULL)
//        {
//            while ((*d_tail = spAtomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//            {
//                if ((data->flag[(*dest)->offset].v == 0)) { break; }
//            }
//
//            if (*d_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//            {
//                while (((*s_tail = spAtomicAdd(g_s_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE))
//                {
//                    if (P_GET_FLAG((*src)->data, *s_tail).v == tag.v) { return SP_MP_SUCCESS; }
//                }
//            }
//        }
//        spParallelSyncThreads();
//        if ((*dest) == NULL)
//        {
//            if (spParallelThreadNum() == 0)
//            {
//                *dest = *pool;
//                *pool = (*pool)->next;
//                if (*dest != NULL) { (*dest)->next = NULL; }
//                *g_d_tail = 0;
//            }
//        }
//        else if (*d_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
//        {
//            dest = &((*dest)->next);
//            if (spParallelThreadNum() == 0) { *g_d_tail = 0; }
//        }
//        else if (*s_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
//        {
//            src = &((*src)->next);
//            if (spParallelThreadNum() == 0) { *g_s_tail = 0; }
//        }
//
//        spParallelSyncThreads();
//
//        if (*dest == NULL) { return SP_MP_ERROR_POOL_IS_OVERFLOW; }
//    }
//
//    return SP_MP_FINISHED;
//
//}

