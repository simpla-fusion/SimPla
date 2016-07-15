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
#include "../../../../../usr/local/cuda/include/cuda_runtime_api.h"

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

    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, int, flag);
    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, Real, rx);
    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, Real, ry);
    ADD_PARTICLE_ATTRIBUTE((*sp), struct spParticlePoint_s, Real, rz);

}

MC_GLOBAL void spParticleDeployKernel(spParticlePage *pg, size_type num_of_pages, size_type size_of_page_in_byte)
{
    size_type offset = spParallelBlockNum() * spParallelNumOfThreads() + spParallelThreadNum();
    for (size_type pos = offset; pos < offset + spParallelNumOfThreads() && pos < num_of_pages;
         pos += spParallelNumOfThreads())
    {
        ((spParticlePage *) ((byte_type *) (pg) + pos * size_of_page_in_byte))->next =
            (struct spParticlePage_s *) ((byte_type *) (pg) + (pos + 1) * size_of_page_in_byte);

        ((spParticlePage *) ((byte_type *) (pg) + pos * size_of_page_in_byte))->id.v = 0;
    }
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

void spParticleDeploy(spParticle *sp, int PIC)
{
    size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, sp->iform/*volume*/);

    size_type num_page_per_cel = (size_type) (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2 + 1;

    sp->number_of_pages = number_of_cell * num_page_per_cel;

    sp->entity_size_in_byte = (size_type) (sp->attrs[sp->num_of_attrs - 1].size_in_byte
        + sp->attrs[sp->num_of_attrs - 1].offset);

    sp->page_size_in_byte = sizeof(struct spParticlePage_s) + sp->entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;

    spParallelDeviceMalloc((void **) (&(sp->m_pages_)), sp->number_of_pages * sp->page_size_in_byte);

    spParallelDeviceMalloc((void **) (&(sp->m_page_pool_)), sizeof(spPage *));

    spParallelMemcpy((void *) (sp->m_page_pool_), (void const *) &(sp->m_pages_), sizeof(spPage *));

    spParallelDeviceMalloc((void **) (&(sp->m_buckets_)), sizeof(spPage *) * number_of_cell);

    spParallelMemset((void *) ((sp->m_buckets_)), 0x0, sizeof(spPage *) * number_of_cell);

    LOAD_KERNEL(spParticleDeployKernel, sp->m->dims, NUMBER_OF_THREADS_PER_BLOCK, sp->m_pages_, sp->number_of_pages,
                sp->page_size_in_byte);

    spParallelDeviceSync();        //wait for iteration to finish

    DONE
}

void spParticleDestroy(spParticle **sp)
{

    spParallelDeviceFree((void **) &((*sp)->m_buckets_));
    spParallelDeviceFree((void **) &((*sp)->m_page_pool_));
    spParallelDeviceFree((void **) &((*sp)->m_pages_));

    free(*sp);
    *sp = NULL;
}

struct spParticleAttrEntity_s *spParticleAddAttribute(spParticle *pg, char const *name, int type_tag,
                                                      size_type size_in_byte, size_type offset)
{
    struct spParticleAttrEntity_s *res = &(pg->attrs[pg->num_of_attrs]);
    strcpy(res->name, name);
    res->type_tag = type_tag;
    res->size_in_byte = size_in_byte;
    if (offset == size_type(-1))
    {
        if (pg->num_of_attrs == 0) { offset = 0; }
        else
        {
            offset = (pg->attrs[pg->num_of_attrs - 1].offset
                + pg->attrs[pg->num_of_attrs - 1].size_in_byte);
        }
    }
    res->offset = offset;
    ++pg->num_of_attrs;
    return res;
}

enum { SP_COPY_OUT_ONLY_DATA = 1, SP_COPY_OUT_WHOLE_PAGE = 2 };

MC_GLOBAL void spParticleCountKernel(dim3 dim, dim3 offset, dim3 count, spParticlePage **buckets,
                                     size_type *page_count_device)
{
    dim3 idx = spParallelBlockIdx();

    if (count.x > idx.x && count.y > idx.y && count.z > idx.z)
    {
        size_type pos = idx.x + offset.x + (idx.y + offset.y + (idx.z + offset.z) * dim.y) * dim.x;
        size_type pos2 = idx.x + (idx.y + idx.z * count.y) * count.x;

        size_type pg_count = 0;

        spParticlePage *pg = buckets[pos];

        while (pg != NULL)
        {
            ++pg_count;
            pg = pg->next;
        }

        page_count_device[pos2] = pg_count;
    }
}

MC_GLOBAL void spParticleDumpKernel(const size_type *offset, spParticlePage **buckets, void **page_ptr_device)
{
    size_type pos = offset[spParallelBlockNum()];

    spParticlePage *pg = buckets[spParallelBlockNum()];

    while (pg != NULL)
    {
        page_ptr_device[pos] = pg;
        ++pos;
        pg = pg->next;
    }
}

MC_GLOBAL void spParticlePopPageKernel(spParticlePage **pool, size_type num, spParticlePage **page_ptr_device)
{


    if (spParallelBlockNum() == 0 && spParallelThreadNum() == 0)
    {
        for (int i = 0; i < num; ++i)
        {
            page_ptr_device[i] = (spParticlePage *) spPageAtomicPop((spPage **) pool);
        }
    }
}
void spParticlePopPages(spParticle const *sp, int num, spParticlePage ***page_ptr_host)
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
size_type spParticleGetPagePointer(spParticle const *sp, dim3 lower, dim3 upper, spParticlePage ***page_ptr_host)
{

    dim3 count;
    count.x = (upper.x - lower.x);
    count.y = (upper.y - lower.y);
    count.z = (upper.z - lower.z);

    size_type number_of_cell = count.x * count.y * count.z;

    if (number_of_cell == 0)
    {
        *page_ptr_host = NULL;
        return 0;
    }

    size_type num_of_pages = 0;
    size_type *page_count_device;


    spParallelDeviceMalloc((void **) (&page_count_device), number_of_cell * sizeof(size_type));

    LOAD_KERNEL(spParticleCountKernel, count, 1, sp->m->dims, lower, count, sp->m_buckets_, page_count_device);

    size_type *page_count_host;

    spParallelHostMalloc((void **) (&page_count_host), number_of_cell * sizeof(size_type));

    spParallelMemcpy((void *) (page_count_host), (void *) (page_count_device), number_of_cell * sizeof(size_type));


    for (int i = 0; i < number_of_cell; ++i)
    {
        size_type old = num_of_pages;
        num_of_pages += page_count_host[i];
        page_count_host[i] = old;
    }
    spParallelMemcpy((void *) (page_count_device), (void *) (page_count_host), number_of_cell * sizeof(size_type));

    spParallelHostFree((void **) &page_count_host);
    /*****************************************************************************************/

    void **page_data_ptr_device;


    spParallelDeviceMalloc((void **) (&page_data_ptr_device), sp->number_of_pages * sizeof(spPage *));


    LOAD_KERNEL(spParticleDumpKernel,
                sp->m->dims, 1,
                page_count_device,
                sp->m_buckets_,
                page_data_ptr_device);

    spParallelDeviceFree((void **) &page_count_device);

    spParallelHostMalloc((void **) (page_ptr_host), sp->number_of_pages * sizeof(spPage *));

    spParallelMemcpy((void *) (*page_ptr_host),
                     (void *) (page_data_ptr_device),
                     sp->number_of_pages * sizeof(spPage *));

    spParallelDeviceFree((void **) &page_data_ptr_device);


    return num_of_pages;
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


    size_type num_of_pages = 0;
    spParticlePage **page_data_ptr_host = NULL;

    num_of_pages = spParticleGetPagePointer(sp, sp->m->i_lower, sp->m->i_upper, &page_data_ptr_host);

    if (page_data_ptr_host == NULL) { return; }

    spParticlePage *write_buffer = NULL;

    spParallelHostMalloc((void **) &write_buffer, sp->page_size_in_byte);

    for (int i = 0; i < num_of_pages; ++i)
    {
        spParallelMemcpy((void *) (write_buffer),
                         (void *) (page_data_ptr_host[i]),
                         sp->page_size_in_byte);


        size_type num = SP_NUMBER_OF_ENTITIES_IN_PAGE;

        Real3 x0 = spMeshPoint(sp->m, write_buffer->id);

        for (int j = 0, je = SP_NUMBER_OF_ENTITIES_IN_PAGE; j < je; ++j)
        {
            P_GET(write_buffer->data, struct spParticlePoint_s, Real, rx, j) += x0.x;
            P_GET(write_buffer->data, struct spParticlePoint_s, Real, ry, j) += x0.y;
            P_GET(write_buffer->data, struct spParticlePoint_s, Real, rz, j) += x0.z;
        }

//        for (int j = 0, je = sp->num_of_attrs; j < je; ++j)
//        {
//            size_type offset_in_byte = sp->attrs[j].offset * SP_NUMBER_OF_ENTITIES_IN_PAGE;
//
//            spIOStreamWriteSimple(os,
//                                  sp->attrs[i].name,
//                                  sp->attrs[i].type_tag,
//                                  (void *) (write_buffer->data + offset_in_byte),
//                                  1,
//                                  &num,
//                                  NULL,
//                                  &num,
//                                  SP_FILE_APPEND);
//        }
    }

    spParallelHostFree((void **) &write_buffer);

    spParallelHostFree((void **) &page_data_ptr_host);

    spIOStreamOpen(os, curr_path);

}

void spParticleRead(spParticle *f, char const url[], int flag)
{

}

struct spParticleSyncStatus_s
{
    int num_of_neighbour;
    int num_of_send;
    int count_of_eof;

    spParticlePage **send_pages_ptr;
    int ***send_pages;
    MPI_Request *send_requests;

    int num_of_recv;
    MPI_Request *recv_requests;

    spParticlePage **recv_buff;
    int *recv_count;
};

void spParticleSyncStatusCreate(struct spParticleSyncStatus_s **p, int num_of_send, int num_of_recv)
{
    *p = (struct spParticleSyncStatus_s *) malloc(sizeof(struct spParticleSyncStatus_s));
    (*p)->num_of_send = num_of_send;
    (*p)->send_requests = (MPI_Request *) malloc(sizeof(MPI_Request) * (num_of_send + 1));

    (*p)->num_of_recv = num_of_recv;
    (*p)->recv_requests = (MPI_Request *) malloc(sizeof(MPI_Request) * (num_of_recv + 1));

    (*p)->recv_buff = (spParticlePage **) malloc(sizeof(spParticlePage *) * num_of_recv);
    (*p)->recv_count = (int *) malloc(sizeof(int) * num_of_recv);

    (*p)->count_of_eof = 0;

}

void spParticleSyncStatusDestroy(struct spParticleSyncStatus_s **p)
{
    spParallelHostFree((void **) &((*p)->recv_requests));
    spParallelHostFree((void **) &((*p)->send_requests));
    spParallelHostFree((void **) &((*p)->recv_buff));
    spParallelHostFree((void **) &((*p)->recv_count));
}

void spParticleSyncStart(spParticle *sp, struct spParticleSyncStatus_s **reqs)
{

    size_type num_of_pages = 0;

    spParticlePage **page_ptr_host;
    num_of_pages = spParticleGetPagePointer(sp, sp->m->i_lower, sp->m->i_upper, &page_ptr_host);
    if (num_of_pages <= 0)
    {
        *reqs = NULL;
        return;
    }

    spParticleSyncStatusCreate(reqs, (int) num_of_pages, 6);

    int offset[3] = {1, 0, 0};

    int dest = 0, send_tag = 0, recv_tag;

    spMPIMakeSendRecvTag(sp->id, offset, &dest, &send_tag, &recv_tag);

    for (int i = 0; i < num_of_pages; ++i)
    {

        MPI_ERROR(MPI_Isend(page_ptr_host[i],
                            (int) (sp->page_size_in_byte),
                            MPI_BYTE,
                            dest,
                            send_tag,
                            spMPIComm(),
                            &((*reqs)->send_requests[i])));
    }
    spParallelHostFree((void **) &page_ptr_host);

    MPI_ERROR(MPI_Isend(NULL, 0, MPI_BYTE, dest, send_tag, MPI_COMM_WORLD, &((*reqs)->send_requests[num_of_pages])));

    /*********************************************************************************************************/

    spParticlePopPages(sp, (*reqs)->num_of_recv, &((*reqs)->recv_buff));

    for (int i = 0; i < (*reqs)->num_of_recv; ++i)
    {
        MPI_ERROR(MPI_Irecv((void *) ((*reqs)->recv_buff[i]),
                            (int) (sp->page_size_in_byte),
                            MPI_BYTE,
                            MPI_ANY_SOURCE,
                            MPI_ANY_TAG,
                            MPI_COMM_WORLD,
                            &((*reqs)->recv_requests[i])));
    }

    (*reqs)->count_of_eof = 0;

}

void spParticleSyncEnd(spParticle *sp, struct spParticleSyncStatus_s **reqs)
{

    while (1)
    {
        for (int count_of_recv = 0;
             (*reqs)->count_of_eof < (*reqs)->num_of_neighbour && count_of_recv < (*reqs)->num_of_recv;
             ++count_of_recv)
        {
            int idx = 0;

            MPI_Status recv_status;

            MPI_Waitany((*reqs)->num_of_send, (*reqs)->recv_requests, &idx, &recv_status);

            CHECK_INT(idx);

            MPI_ERROR(MPI_Get_count(&recv_status, MPI_BYTE, &((*reqs)->recv_count[idx])));
            CHECK_INT((*reqs)->recv_count[idx]);
            if ((*reqs)->recv_count[idx] == 0) { ++((*reqs)->count_of_eof); }
        }

        if ((*reqs)->count_of_eof >= (*reqs)->num_of_neighbour) { break; }
        else
        {
            for (int i = 0; i < (*reqs)->num_of_recv; ++i)
            {
                MPI_ERROR(MPI_Irecv((void *) ((*reqs)->recv_buff[i]),
                                    (int) (sp->page_size_in_byte),
                                    MPI_BYTE,
                                    MPI_ANY_SOURCE,
                                    MPI_ANY_TAG,
                                    MPI_COMM_WORLD,
                                    &((*reqs)->recv_requests[i])));
            }
        }
    }

    for (int i = 0; i < (*reqs)->num_of_recv; ++i) { MPI_Cancel(&((*reqs)->recv_requests[i])); }

    MPI_Waitall((*reqs)->num_of_send, (*reqs)->send_requests, MPI_STATUS_IGNORE);

    spParticleSyncStatusDestroy(reqs);

}

void spParticleSync(spParticle *sp)
{
    struct spParticleSyncStatus_s *reqs;

    spParticleSyncStart(sp, &reqs);

    spParticleSyncEnd(sp, &reqs);

    spParticleSyncStatusDestroy(&reqs);

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
//		size_type entity_size_in_byte, int MASK, MeshEntityId tag)
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

MC_DEVICE int spParticleMapAndPack(spParticlePage **dest,
                                   spParticlePage **src,
                                   int *d_tail,
                                   int *g_d_tail,
                                   int *s_tail,
                                   int *g_s_tail,
                                   spParticlePage **pool,
                                   MeshEntityId tag)
{

    while (*src != NULL)
    {
        if ((*dest) != NULL)
        {
            while ((*d_tail = spAtomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
            {
                if ((P_GET_FLAG((*dest)->data, *d_tail).v == 0)) { break; }
            }

            if (*d_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
            {
                while (((*s_tail = spAtomicAdd(g_s_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE))
                {
                    if (P_GET_FLAG((*src)->data, *s_tail).v == tag.v) { return SP_MP_SUCCESS; }
                }
            }
        }
        spParallelSyncThreads();
        if ((*dest) == NULL)
        {
            if (spParallelThreadNum() == 0)
            {
                *dest = *pool;
                *pool = (*pool)->next;
                if (*dest != NULL) { (*dest)->next = NULL; }
                *g_d_tail = 0;
            }
        }
        else if (*d_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
        {
            dest = &((*dest)->next);
            if (spParallelThreadNum() == 0) { *g_d_tail = 0; }
        }
        else if (*s_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
        {
            src = &((*src)->next);
            if (spParallelThreadNum() == 0) { *g_s_tail = 0; }
        }

        spParallelSyncThreads();

        if (*dest == NULL) { return SP_MP_ERROR_POOL_IS_OVERFLOW; }
    }

    return SP_MP_FINISHED;

}
