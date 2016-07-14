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

void spParticleCreate(const spMesh *mesh, spParticle **sp)
{
    *sp = (spParticle *) malloc(sizeof(spParticle));

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

MC_GLOBAL void spParticleDumpKernel(const size_type *offset, spParticlePage **buckets, void **page_ptr_device,
                                    MeshEntityId *page_id_device, int only_data)
{
    size_type pos = offset[spParallelBlockNum()];

    spParticlePage *pg = buckets[spParallelBlockNum()];

    while (pg != NULL)
    {
        if (only_data == SP_COPY_OUT_ONLY_DATA) { page_ptr_device[pos] = pg->data; } else { page_ptr_device[pos] = pg; }

        dim3 idx = spParallelBlockIdx();
        page_id_device[pos].w = 0;
        page_id_device[pos].x = (int16_t) idx.x;
        page_id_device[pos].y = (int16_t) idx.y;
        page_id_device[pos].z = (int16_t) idx.z;

        ++pos;
        pg = pg->next;
    }
}

size_type spParticleGetDevicePtrOfPage(spParticle const *sp, dim3 offset, dim3 count, MeshEntityId **page_id_host,
                                       void ***page_ptr_host, int only_data)
{
    size_type num_of_pages = 0;

    size_type number_of_cell = (count.x) * (count.y) * (count.z);

    size_type *page_count_device;

    spParallelDeviceMalloc((void **) (&page_count_device), number_of_cell * sizeof(size_type));

    LOAD_KERNEL(spParticleCountKernel, count, 1, sp->m->dims, offset, count, sp->m_buckets_, page_count_device);

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

    MeshEntityId *page_id_device;

    spParallelDeviceMalloc((void **) (&page_data_ptr_device), sp->number_of_pages * sizeof(spPage *));
    spParallelDeviceMalloc((void **) (&page_id_device), sp->number_of_pages * sizeof(MeshEntityId));

    LOAD_KERNEL(spParticleDumpKernel, sp->m->dims, 1, page_count_device, sp->m_buckets_, page_data_ptr_device,
                page_id_device, only_data);

    spParallelDeviceFree((void **) &page_count_device);

    spParallelHostMalloc((void **) (page_ptr_host), sp->number_of_pages * sizeof(spPage *));
    spParallelMemcpy((void *) (*page_ptr_host), (void *) (page_data_ptr_device),
                     sp->number_of_pages * sizeof(spPage *));
    spParallelDeviceFree((void **) &page_data_ptr_device);

    spParallelHostMalloc((void **) (page_id_host), sp->number_of_pages * sizeof(MeshEntityId));
    spParallelMemcpy((void *) (*page_id_host), (void *) (page_id_device), sp->number_of_pages * sizeof(MeshEntityId));
    spParallelDeviceFree((void **) &page_id_device);

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
    size_type num_of_pages = 0;
    void **page_data_ptr_host;
    MeshEntityId *page_id_host;

    char curr_path[2048];
    char new_path[2048];
    strcpy(new_path, name);
    new_path[strlen(name)] = '/';
    new_path[strlen(name) + 1] = '\0';

    spIOStreamPWD(os, curr_path);
    spIOStreamOpen(os, new_path);
    dim3 count;
    count.x = sp->m->i_upper.x - sp->m->i_lower.x;
    count.y = sp->m->i_upper.y - sp->m->i_lower.y;
    count.z = sp->m->i_upper.z - sp->m->i_lower.z;

    num_of_pages = spParticleGetDevicePtrOfPage(sp, sp->m->i_lower, count, &page_id_host,
                                                &page_data_ptr_host, SP_COPY_OUT_ONLY_DATA);

    size_type f_dims[2];

    f_dims[0] = num_of_pages;
    f_dims[1] = SP_NUMBER_OF_ENTITIES_IN_PAGE;

    spIOStreamWriteSimple(os, "MeshId", (int) SP_TYPE_MeshEntityId, page_id_host, 1, f_dims, NULL, f_dims, flag);

    for (int i = 0; i < sp->num_of_attrs; ++i)
    {
        void *d;
        size_type page_size_in_byte = sp->attrs[i].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;
        size_type offset_in_byte = sp->attrs[i].offset * SP_NUMBER_OF_ENTITIES_IN_PAGE;
        spParallelHostMalloc(&d, num_of_pages * page_size_in_byte);

        for (int j = 0; j < num_of_pages; ++j)
        {
            spParallelMemcpy((void *) ((byte_type *) (d) + j * page_size_in_byte),
                             (void *) ((byte_type *) (page_data_ptr_host[j]) + offset_in_byte), page_size_in_byte);
        }

        spIOStreamWriteSimple(os, sp->attrs[i].name, sp->attrs[i].type_tag, d, 2, f_dims, NULL, f_dims, flag);

        spParallelHostFree(&d);
    }

    spParallelHostFree((void **) &page_id_host);
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
    (*p)->send_requests = (MPI_Request *) malloc(sizeof(MPI_Request) * (num_of_recv + 1));

    *((*p)->recv_buff) = (spParticlePage *) malloc(sizeof(spParticlePage *) * num_of_recv);
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
    void **page_ptr_host;
    MeshEntityId *page_id_host;
    dim3 count;
    count.x = sp->m->i_lower.x - sp->m->offset.x;
    count.y = sp->m->i_upper.y - sp->m->i_lower.y;
    count.z = sp->m->i_upper.z - sp->m->i_lower.z;
    CHECK("START SYNC");
    num_of_pages = spParticleGetDevicePtrOfPage(sp, sp->m->i_lower, count, &page_id_host,
                                                &page_ptr_host, SP_COPY_OUT_WHOLE_PAGE);
    CHECK("START SYNC");
    spParticleSyncStatusCreate(reqs, (int) num_of_pages, 6);
    CHECK("START SYNC");


    int dest = 0, tag = 0;
    for (int i = 0; i < num_of_pages; ++i)
    {
        MPI_ERROR(MPI_Isend(page_ptr_host[i],
                            (int) (sp->page_size_in_byte),
                            MPI_BYTE,
                            dest,
                            tag,
                            MPI_COMM_WORLD,
                            &((*reqs)->send_requests[i])));
    }
    CHECK("START SYNC");

    MPI_ERROR(MPI_Isend(NULL, 0, MPI_BYTE, dest, tag, MPI_COMM_WORLD, &((*reqs)->send_requests[num_of_pages])));
    CHECK("START SYNC");


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
    CHECK("START SYNC");

    (*reqs)->count_of_eof = 0;
    CHECK("START SYNC");

    spParallelHostFree((void **) &page_ptr_host);
    spParallelHostFree((void **) &page_id_host);
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

            MPI_ERROR(MPI_Get_count(&recv_status, MPI_BYTE, &((*reqs)->recv_count[idx])));

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
    CHECK("START SYNC");
    spParticleSyncStart(sp, &reqs);
    CHECK("STOP SYNC");
    spParticleSyncEnd(sp, &reqs);
    CHECK("DESTROY STATUS");
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
