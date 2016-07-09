/*
 * spParticleKernel.cu
 *
 *  Created on: 2016年7月6日
 *      Author: salmon
 */
#include <assert.h>
#include "spParticle.h"
#include "spPage.h"
#include "spMesh.h"
#include "spParallel.h"
//
//MC_GLOBAL void spParticleInitializeKernel(int num_of_pages, spPage *pg, void *data, size_type entity_size_in_byte,
//                                          size_type page_size_in_byte)
//{
//    for (int pos = spParallelThreadId(); pos < num_of_pages; pos += spParallelNumOfThreads())
//    {
//        pg[pos].next = &(pg[pos + 1]);
//        pg[pos].data = data + entity_size_in_byte * SP_NUMBER_OF_ENTITIES_IN_PAGE * pos;
//    }
//
//}
//
//void spParticleInitialize(spParticle *sp)
//{
//    //@formatter:off
//	spParticleInitializeKernel<<<sp->m->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets, sp->m_pages_holder, sp->data,
//			sp->entity_size_in_byte, sp->page_size_in_byte);
//	//@formatter:on
//    spParallelThreadSync();        //wait for iteration to finish
//
//}

MC_DEVICE int spParticleMapAndPack(spPage **dest, spPage const **src, int *d_tail, int *g_d_tail, int *s_tail,
		int *g_s_tail,
//                               spPage **pool,
		MeshEntityId tag)
{

//	while (*src != NULL)
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
//
//			if (*d_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//			{
//				while (((*s_tail = spAtomicAdd(g_s_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE))
//				{
//					if (P_GET_FLAG((*src)->data, *s_tail).v == tag.v)
//					{
//						return SP_MP_SUCCESS;
//					}
//				}
//			}
//		}
//		spParallelSyncThreads();
//		if ((*dest) == NULL)
//		{
////            if (spParallelThreadId() == 0)
////            {
////                *dest = *pool;
////                *pool = (*pool)->next;
////                if (*dest != NULL) { (*dest)->next = NULL; }
////                *g_d_tail = 0;
////            }
//		}
//		else if (*d_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
//		{
//			dest = &((*dest)->next);
//			if (spParallelThreadNum() == 0)
//			{
//				*g_d_tail = 0;
//			}
//		}
//		else if (*s_tail >= SP_NUMBER_OF_ENTITIES_IN_PAGE)
//		{
//			src = (spPage const **) &((*src)->next);
//			if (spParallelThreadNum() == 0)
//			{
//				*g_s_tail = 0;
//			}
//		}
//
//		spParallelSyncThreads();
//
//		if (*dest == NULL)
//		{
//			return SP_MP_ERROR_POOL_IS_OVERFLOW;
//		}
//	}

	return SP_MP_FINISHED;

}
