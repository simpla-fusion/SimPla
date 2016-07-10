/**
 * spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
#include <stdlib.h>
#include <string.h>
#include <assert.h>
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
	for (size_type pos = offset; pos < offset + spParallelNumOfThreads() && pos < num_of_pages; pos +=
	spParallelNumOfThreads())
	{
		((struct spParticlePage_s *) ((byte_type*) (pg) + pos * size_of_page_in_byte))->next =
				(struct spParticlePage_s *) ((byte_type*) (pg) + (pos + 1) * size_of_page_in_byte);
		((spParticlePage *) ((byte_type*) (pg) + pos * size_of_page_in_byte))->id.v = 0;
	}
}

MC_GLOBAL void spParticleTestAtomicPageOp(spParticlePage **pg)
{
	MC_SHARED spPage * p;
	spParallelSyncThreads();
	if (spParallelThreadNum() == 0)
	{
		p = (spPage*) (*pg);
	}
	spParallelSyncThreads();
	spPage * res = spPageAtomicPop(&p);
	CUDA_CHECK(res);
	if (spParallelThreadNum() == 0)
	{
		(*pg) = (spParticlePage*) p;
	}

}
void spParticleDeploy(spParticle *sp, int PIC)
{
	size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, sp->iform/*volume*/);

	size_type num_page_per_cel = (size_type) (PIC * 3 / SP_NUMBER_OF_ENTITIES_IN_PAGE) / 2 + 1;
	sp->number_of_pages = number_of_cell * num_page_per_cel;

	sp->entity_size_in_byte = (size_type) (sp->attrs[sp->num_of_attrs - 1].size_in_byte
			+ sp->attrs[sp->num_of_attrs - 1].offsetof);

	sp->page_size_in_byte = sizeof(struct spParticlePage_s) + sp->entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;

	spParallelDeviceMalloc((void **) (&(sp->m_pages_)), sp->number_of_pages * sp->page_size_in_byte);

	spParallelDeviceMalloc((void **) (&(sp->m_page_pool_)), sizeof(spPage*));

	spParallelMemcpy((void *) (sp->m_page_pool_), (void const*) &(sp->m_pages_), sizeof(spPage*));

	spParallelDeviceMalloc((void **) (&(sp->m_buckets_)), sizeof(spPage*) * number_of_cell);

	spParallelMemset((void *) ((sp->m_buckets_)), 0x0, sizeof(spPage*) * number_of_cell);

	spParticleDeployKernel<<<sp->m->dims,NUMBER_OF_THREADS_PER_BLOCK>>>(sp->m_pages_, sp->number_of_pages,sp->page_size_in_byte );

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
		size_type size_in_byte, int offset)
{
	struct spParticleAttrEntity_s *res = &(pg->attrs[pg->num_of_attrs]);
	strcpy(res->name, name);
	res->type_tag = type_tag;
	res->size_in_byte = size_in_byte;
	if (offset == -1)
	{
		if (pg->num_of_attrs == 0)
		{
			offset = 0;
		}
		else
		{
			offset = pg->attrs[pg->num_of_attrs - 1].offsetof + pg->attrs[pg->num_of_attrs - 1].size_in_byte;
		}
	}
	res->offsetof = offset;
	++pg->num_of_attrs;
	return res;
}

//void *spParticleGetAttribute(spParticle *sp, char const *name)
//{
//	void *res = NULL;
//	for (int i = 0; i < sp->number_of_attrs; ++i)
//	{
//		if (strcmp(sp->attrs[i].name, name) > 0)
//		{
//			res = (byte_type*) (sp->data) + sp->attrs[i].offsetof * sp->number_of_pages * SP_NUMBER_OF_ENTITIES_IN_PAGE;
//		}
//	}
//
//	return res;
//}

MC_GLOBAL void spParticleCountKernel(dim3 dim, spParticlePage **buckets, int * res)
{

	dim3 idx = spParallelBlockIdx();

	if (dim.x > idx.x && dim.y > idx.y && dim.z > idx.z)
	{
		size_type pos = idx.x + (idx.y + idx.z * dim.y) * dim.y;

		int count = 0;

		spParticlePage * pg = buckets[pos];

		while (pg != NULL)
		{
			++count;
			pg = pg->next;
		}

		res[pos] = count;
	}
}
MC_GLOBAL void spParticleDumpKernel(int const * offset, spParticlePage **buckets, void** page_data_ptr_device,
		MeshEntityId * page_id_device, int * res)
{
	int count = 0;

	int pos = offset[spParallelBlockNum()];

	spParticlePage * pg = buckets[spParallelBlockNum()];

	while (pg != NULL)
	{
		page_data_ptr_device[pos] = pg->data;
		dim3 idx = spParallelBlockIdx();
		page_id_device[pos].w = 0;
		page_id_device[pos].x = idx.x;
		page_id_device[pos].y = idx.y;
		page_id_device[pos].z = idx.z;

		++pos;
		pg = pg->next;
	}
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

	size_type number_of_cell = spMeshGetNumberOfEntity(sp->m, sp->iform/*volume*/);

	int *page_count_device;
	spParallelDeviceMalloc((void**) (&page_count_device), number_of_cell * sizeof(int));

	spParticleCountKernel<<<sp->m->dims,1>>>( sp->m->dims, sp->m_buckets_, page_count_device);

	size_type num_of_pages = 0;
	int *page_count_host;

	spParallelHostMalloc((void**) (&page_count_host), number_of_cell * sizeof(int));

	spParallelMemcpy((void*) (page_count_host), (void*) (page_count_device), number_of_cell * sizeof(int));

	for (int i = 0; i < number_of_cell; ++i)
	{
		size_type old = num_of_pages;
		num_of_pages += page_count_host[i];
		page_count_host[i] = old;
	}
	spParallelMemcpy((void*) (page_count_device), (void*) (page_count_host), number_of_cell * sizeof(int));

	spParallelHostFree((void**) &page_count_host);
	/*****************************************************************************************/

	void** page_data_ptr_device;
	void** page_data_ptr_host;
	MeshEntityId * page_id_device;
	MeshEntityId * page_id_host;

	spParallelDeviceMalloc((void**) (&page_data_ptr_device), sp->number_of_pages * sizeof(spPage*));
	spParallelDeviceMalloc((void**) (&page_id_device), sp->number_of_pages * sizeof(MeshEntityId));

	spParticleDumpKernel<<<sp->m->dims,1>>>( page_count_device ,sp->m_buckets_,page_data_ptr_device, page_id_device, page_count_device);

	spParallelDeviceFree((void**) &page_count_device);

	spParallelHostMalloc((void**) (&page_data_ptr_host), sp->number_of_pages * sizeof(spPage*));
	spParallelMemcpy((void*) (page_data_ptr_host), (void*) (page_data_ptr_device),
			sp->number_of_pages * sizeof(spPage*));
	spParallelDeviceFree((void**) &page_data_ptr_device);

	spParallelHostMalloc((void**) (&page_id_host), sp->number_of_pages * sizeof(MeshEntityId));
	spParallelMemcpy((void*) (page_id_host), (void*) (page_id_device), sp->number_of_pages * sizeof(MeshEntityId));
	spParallelDeviceFree((void**) &page_id_device);

	size_type dims[2];

	dims[0] = num_of_pages;
	dims[1] = SP_NUMBER_OF_ENTITIES_IN_PAGE;

	spIOStreamWriteSimple(os, "MeshId", (int) SP_TYPE_MeshEntityId, page_id_host, 1, dims, NULL, dims, (int) flag);

	for (int i = 0; i < sp->num_of_attrs; ++i)
	{
		void *d;
		size_type page_size_in_byte = sp->attrs[i].size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE;
		size_type offset_in_byte = sp->attrs[i].offsetof * SP_NUMBER_OF_ENTITIES_IN_PAGE;
		spParallelHostMalloc(&d, num_of_pages * page_size_in_byte);

		for (int j = 0; j < num_of_pages; ++j)
		{
			spParallelMemcpy((void*) ((byte_type*) (d) + j * page_size_in_byte),
					(page_data_ptr_host[j] + offset_in_byte), page_size_in_byte);
		}

		spIOStreamWriteSimple(os, sp->attrs[i].name, sp->attrs[i].type_tag, d, 2, dims, NULL, dims, (int) flag);

		spParallelHostFree(&d);
	}

	spParallelHostFree((void**) &page_id_host);
	spParallelHostFree((void**) &page_data_ptr_host);

	spIOStreamOpen(os, curr_path);

}

void spParticleRead(spParticle *f, char const url[], int flag)
{

}

void spParticleSync(spParticle *f)
{
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

MC_DEVICE int spParticleMapAndPack(spParticlePage **dest, spParticlePage **src, int *d_tail, int *g_d_tail, int *s_tail,
		int *g_s_tail, spParticlePage **pool, MeshEntityId tag)
{

	while (*src != NULL)
	{
		if ((*dest) != NULL)
		{
			while ((*d_tail = spAtomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
			{
				if ((P_GET_FLAG((*dest)->data, *d_tail).v == 0))
				{
					break;
				}
			}

			if (*d_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
			{
				while (((*s_tail = spAtomicAdd(g_s_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE))
				{
					if (P_GET_FLAG((*src)->data, *s_tail).v == tag.v)
					{
						return SP_MP_SUCCESS;
					}
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
				if (*dest != NULL)
				{
					(*dest)->next = NULL;
				}
				*g_d_tail = 0;
			}
		}
		else if (*d_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
		{
			dest = &((*dest)->next);
			if (spParallelThreadNum() == 0)
			{
				*g_d_tail = 0;
			}
		}
		else if (*s_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
		{
			src = &((*src)->next);
			if (spParallelThreadNum() == 0)
			{
				*g_s_tail = 0;
			}
		}

		spParallelSyncThreads();

		if (*dest == NULL)
		{
			return SP_MP_ERROR_POOL_IS_OVERFLOW;
		}
	}

	return SP_MP_FINISHED;

}
