/**
 * @file spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdlib.h>


#include "sp_lite_def.h"
#include "spParallel.h"
#include "spParallelCUDA.h"

#include "spMesh.h"
#include "spPage.h"
#include "spParticle.h"
#include "spPage.cu.h"

#ifdef __cplusplus
}
#endif

__global__ void spParticlePushPageToFieldKernel(spParticlePage **base,
                                                spParticlePage **pool,
                                                dim3 dims,
                                                dim3 offset,
                                                size_type const *num_of_pages,
                                                size_type default_num_page)
{
    size_type pos = offset.x + THREAD_X + (THREAD_Y + THREAD_Z * dims.y) * dims.x;

    spPageLinkResize((spPage **) &base[pos],
                     (spPage **) pool,
                     (num_of_pages == NULL) ? default_num_page : num_of_pages[pos]);
}
int
spParticlePushPageToField(spParticlePage **b, // device
                          spParticlePage **pool,// device
                          size_type const *shape,
                          size_type const *lower,
                          size_type const *upper,
                          size_type const *num_of_pages, // device or null
                          size_type default_num_page)
{

    dim3 count;
    count.x = (unsigned int) (upper[0] - lower[0]);
    count.y = (unsigned int) (upper[1] - lower[1]);
    count.z = (unsigned int) (upper[2] - lower[2]);

    LOAD_KERNEL(
        spParticlePushPageToFieldKernel,
        count,
        1,
        b,
        pool,
        sizeType2Dim3(shape),
        sizeType2Dim3(lower),
        num_of_pages,
        default_num_page);
    return SP_SUCCESS;
}
void spParticleInitialize(spParticle *sp)
{

}

//__global__ void spParticleDeployKernel(spParticlePage **bucket, spParticlePage *pg, size_type num_of_pages)
//{
////    size_type offset = spParallelBlockNum() * spParallelNumOfThreads() + spParallelThreadNum();
////    for (size_type pos = offset; pos < offset + spParallelNumOfThreads() && pos < num_of_pages;
////         pos += spParallelNumOfThreads())
////    {
////        pg[pos].next = &(pg[pos + 1]);
////        pg[pos].offset = pos * SP_NUMBER_OF_ENTITIES_IN_PAGE;
////    }
////
////    if (spParallelThreadNum() == 0) { bucket[spParallelBlockNum()] = NULL; }
//}

//__global__ void spParticlePageExpandKernel(dim3 dims, dim3 lower,
//                                           spParticlePage **buckets,
//                                           int *out_offset,
//                                           int *num)
//{
//
//    MeshEntityId id;
//
//    id.x = blockIdx.x + lower.x;
//    id.y = blockIdx.y + lower.y;
//    id.z = blockIdx.z + lower.z;
//    id.w = 0;
//    spParticlePage *pg = buckets[id.x + (id.y + id.z * dims.y) * dims.x];
//
//    id.x = (id.x << 1) + 1;
//    id.y = (id.y << 1) + 1;
//    id.z = (id.z << 1) + 1;
//
//    while (pg != NULL)
//    {
//        int s = atomicAdd(num, 1);
//        out_offset[s] = (int) (pg->offset);
//        pg = pg->next;
//    }
//
//}
//size_type spParticlePageExpand(spParticle *sp,
//                               size_type const *lower,
//                               size_type const *upper,
//                               size_type max_num_of_pages,
//                               int **out_offset_host)
//{
//
//    size_type count[3];
//
//    for (int i = 0; i < 3; ++i) { count[i] = (upper[i] - lower[i]); }
//
//    size_type num_of_cell = count[0] * count[1] * count[2];
//
//    if (num_of_cell == 0) { return 0; }
//
//    max_num_of_pages = num_of_cell * 2;
//
//    int *out_offset_device = NULL;
//
//    int *num_of_page_device = NULL;
//
//    spParallelDeviceAlloc((void **) (&num_of_page_device), sizeof(int));
//
//    spParallelMemset((void *) (num_of_page_device), 0, sizeof(int));
//
//
//    spParallelDeviceAlloc((void **) (&out_offset_device), max_num_of_pages * sizeof(int));
//
//    LOAD_KERNEL(spParticlePageExpandKernel,
//                sizeType2Dim3(count), 1,
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                sizeType2Dim3(lower),
//                spParticleBuckets(sp),
//                out_offset_device,
//                num_of_page_device);
//
//    int num_of_page = 0;
//
//    spParallelMemcpy((void *) (&num_of_page), (void *) (num_of_page_device), sizeof(int));
//
//    spParallelDeviceFree((void **) (&num_of_page_device));
//
//
//    if (num_of_page > 0)
//    {
//        spParallelHostAlloc((void **) (out_offset_host), sizeof(int) * num_of_page);
//        spParallelMemcpy((void *) (*out_offset_host), (void *) (out_offset_device), num_of_page * sizeof(int));
//    }
//
//
//    spParallelDeviceFree((void **) (&out_offset_device));
//
//    return (size_type) num_of_page;
//}
//
///*****************************************************************************************/
//
//__global__ void spParticlePopPageKernel(spParticlePage **pool, int num, spParticlePage **page_ptr_device)
//{
//
//
////    if (spParallelBlockNum() == 0 && spParallelThreadNum() == 0)
////    {
////        for (int i = 0; i < num; ++i)
////        {
////            page_ptr_device[i] = (spParticlePage *) spPageAtomicPop((spPage **) pool);
////        }
////    }
//}
//void spParticlePopPages(spParticle *sp, int num, spParticlePage **page_ptr_host, int *pos)
//{
//    spParticlePage **page_data_ptr_device;
//
//
//    spParallelDeviceAlloc((void **) (&page_data_ptr_device), num * sizeof(spPage *));
//
//    LOAD_KERNEL(spParticlePopPageKernel, 1, 1, spParticlePagePool(sp), num, page_data_ptr_device);
//
//    spParallelHostAlloc((void **) (page_ptr_host), num * sizeof(spPage *));
//
//    spParallelMemcpy((void *) (*page_ptr_host),
//                     (void *) (page_data_ptr_device),
//                     num * sizeof(spPage *));
//
//    spParallelDeviceFree((void **) &page_data_ptr_device);
//}

//
//MC_DEVICE int spPageInsert(spPage **dest, spPage **pool, int *d_tail, int *g_d_tail)
//{
//	while (1)
//	{
//		if ((*dest) != NULL)
//		{
//			while ((*d_tail = atomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//			{
//				if ((P_GET_FLAG((*dest)->data, *d_tail).v == 0))
//				{
//					break;
//				}
//			}
//		}
//
//		__syncthreads();
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
//		__syncthreads();
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
//		while ((*d_tail = atomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//		{
//			if ((P_GET_FLAG((*dest)->data, *d_tail).v & MASK == tag & MASK))
//			{
//				return SP_MP_SUCCESS;
//			}
//		}
//
//		__syncthreads();
//
//		dest = &((*dest)->next);
//
//		if (THREAD_ID == 0)
//		{
//			g_d_tail = 0;
//		}
//
//		__syncthreads();
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
//	__syncthreads();
//
//	if (spParallelThreadNum() == 0)
//	{
//		g_s_tail = 0;
//		g_d_tail = 0;
//	}
//	__syncthreads();
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
//            while ((*d_tail = atomicAdd(g_d_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//            {
//                if ((data->flag[(*dest)->offset].v == 0)) { break; }
//            }
//
//            if (*d_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
//            {
//                while (((*s_tail = atomicAdd(g_s_tail, 1)) < SP_NUMBER_OF_ENTITIES_IN_PAGE))
//                {
//                    if (P_GET_FLAG((*src)->data, *s_tail).v == tag.v) { return SP_MP_SUCCESS; }
//                }
//            }
//        }
//        __syncthreads();
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
//        __syncthreads();
//
//        if (*dest == NULL) { return SP_MP_ERROR_POOL_IS_OVERFLOW; }
//    }
//
//    return SP_MP_FINISHED;
//
//}

