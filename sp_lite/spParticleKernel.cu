/**
 * @file spParticle.c
 *
 *  Created on: 2016年6月15日
 *      Author: salmon
 */
//
//#ifdef __cplusplus
//extern "C" {
//#endif
//
//#include <assert.h>
//#include <stdlib.h>
//
//
//#include "sp_lite_def.h"
//#include "spParallel.h"
//#include "spParallelCUDA.h"
//
//#include "spMesh.h"
//#include "spPage.h"
//#include "spParticle.h"
//#include "spPage.cu.h"
//
//#ifdef __cplusplus
//}
//#endif
//
//
//__global__ void
//spParticleUpdatePageCountKernel(spParticlePage **b, // device
//                                size_type *page_count,// device
//                                dim3 dims, dim3 offset
//
//)
//{
//    size_type pos = offset.x + THREAD_X + (THREAD_Y + THREAD_Z * dims.y) * dims.x;
//    spParticlePage **p = &b[pos];
//    size_type count = 0;
//    while (*p != NULL)
//    {
//        p = &((*p)->next);
//        ++count;
//    }
//    page_count[pos] = count;
//
//}
//__global__ void
//spParticleResizePageLinkKernel(spParticlePage **base,
//                               spParticlePage **pool,
//                               size_type const *num_of_pages,
//                               dim3 dims,
//                               dim3 offset)
//{
//    size_type pos = offset.x + THREAD_X + (THREAD_Y + THREAD_Z * dims.y) * dims.x;
//
//    spPageLinkResize((spPage **) &base[pos], (spPage **) pool, num_of_pages[pos]);
//}
//
//int
//spParticleResizePageLink(spParticle *sp)
//{
//    size_type shape[3], lower[3], upper[3], count[3];
//
//    spMeshDomain(spParticleMesh(sp), SP_DOMAIN_ALL, lower, upper, shape, NULL);
//
//    count[0] = upper[0] - lower[0];
//    count[1] = upper[1] - lower[1];
//    count[2] = upper[2] - lower[2];
//
//    LOAD_KERNEL(spParticleUpdatePageCountKernel,
//                sizeType2Dim3(count), 1,
//                spParticleBaseField(sp),
//                spParticlePageCount(sp),
//                sizeType2Dim3(shape),
//                sizeType2Dim3(lower)
//    );
//
//
//    spMPIUpdateNdArrayHalo(spParticlePageCount(sp), 3, shape,
//                           lower, NULL, count, NULL, MPI_INT, spMPIComm());
//
//    LOAD_KERNEL(spParticleResizePageLinkKernel,
//                sizeType2Dim3(count), 1,
//                spParticleBaseField(sp),
//                spParticlePagePool(sp),
//                spParticlePageCount(sp),
//                sizeType2Dim3(shape),
//                sizeType2Dim3(lower)
//    );
//
//    return SP_SUCCESS;
//}
//
//__global__ void
//spParticleInitializeKernel(spParticlePage **base,
//                           dim3 dims,
//                           dim3 offset)
//{
//    spParticlePage **p = &base[offset.x + blockIdx.x + (blockIdx.y + blockIdx.z * dims.y) * dims.x];
//
//    int s = threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;
//
//    while (*p != NULL)
//    {
//        (*p)->flag[s].v = 0;
//        (*p)->rx[s] = 1.0;
//        (*p)->ry[s] = 1.0;
//        (*p)->rz[s] = 1.0;
//
//        p = &((*p)->next);
//    }
//
//}
//int spParticleInitialize(spParticle *sp)
//{
//    spParticleResizePageLink(sp);
//
//    size_type shape[3], lower[3], upper[3], count[3];
//
//    spMeshDomain(spParticleMesh(sp), SP_DOMAIN_CENTER, lower, upper, shape, NULL);
//
//    count[0] = upper[0] - lower[0];
//    count[1] = upper[1] - lower[1];
//    count[2] = upper[2] - lower[2];
//    LOAD_KERNEL(spParticleInitializeKernel,
//                sizeType2Dim3(count), spParticleNumOfEntitiesInPage(sp),
//                spParticleBaseField(sp),
//                sizeType2Dim3(shape),
//                sizeType2Dim3(lower)
//    );
//
//    spParticleSync(sp);
//    return SP_SUCCESS;
//}
//__global__ void
//spParticleDumpPageCount(size_type const *page_count, dim3 dims, dim3 offset, size_type *out)
//{
//    out[THREAD_X + (THREAD_Y + THREAD_Z * DIMS_X) * DIMS_Y] =
//        page_count[offset.x + THREAD_X + (offset.y + THREAD_Y + (offset.z + THREAD_Z) * dims.y) * dims.x];
//}
//__global__ void
//spParticleDumpPageOffsetKernel(spParticlePage const **base,
//                               size_type *page_offset,
//                               dim3 dims,
//                               dim3 offset,
//                               spParticlePage const *root,
//                               MeshEntityId *page_id,
//                               size_type *out)
//{
//    size_type it = THREAD_X + (THREAD_Y + THREAD_Z * DIMS_X) * DIMS_Y;
//
//    spParticlePage const **p = &base[offset.x + blockIdx.x + (blockIdx.y + blockIdx.z * dims.y) * dims.x];
//    size_type displ = page_offset[it];
//
//    while (*p != NULL)
//    {
//        out[displ] = size_t(*p) - size_t(root);
//        p = (spParticlePage const **) (&((*p)->next));
//        ++displ;
//    }
//}
//int
//spParticleGetPageOffset(spParticle *sp,
//                        size_type const lower[3],
//                        size_type const upper[3],
//                        size_type *num_of_page,
//                        MeshEntityId **page_id,
//                        size_type **data_displs)
//{
//
//    size_type count[3];
//
//
//    count[0] = upper[0] - lower[0];
//    count[1] = upper[1] - lower[1];
//    count[2] = upper[2] - lower[2];
//
//
//    size_type *page_offset_d;
//
//    size_type *page_offset_h;
//
//    size_type num_of_cell = count[0] * count[1] * count[2];
//
//    spParallelDeviceAlloc(&page_offset_d, sizeof(size_type) * num_of_cell);
//    spParallelHostAlloc(&page_offset_h, sizeof(size_type) * num_of_cell + 1);
//
//    LOAD_KERNEL(spParticleDumpPageCount,
//                sizeType2Dim3(count), 1,
//                spParticlePageCount(sp),
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                sizeType2Dim3(lower),
//                page_offset_d);
//
//    spParallelMemcpy(page_offset_h + 1, page_offset_d, sizeof(size_type) * num_of_cell);
//
//    page_offset_h[0] = 0;
//
//    for (size_type s = 1; s <= num_of_cell; ++s) { page_offset_h[s] += page_offset_h[s - 1]; }
//
//    spParallelMemcpy(page_offset_d, page_offset_h, sizeof(size_type) * num_of_cell);
//
//    *num_of_page = page_offset_h[num_of_cell];
//
//    size_type *disp_d;
//    MeshEntityId *page_id_d;
//
//    spParallelDeviceAlloc(&disp_d, sizeof(size_type) * (*num_of_page));
//    spParallelDeviceAlloc(&page_id_d, sizeof(MeshEntityId) * (*num_of_page));
//
//
//    LOAD_KERNEL(spParticleDumpPageOffsetKernel,
//                sizeType2Dim3(count), 1,
//                (spParticlePage const **) spParticleBaseField(sp),
//                page_offset_d,
//                sizeType2Dim3(spMeshGetShape(spParticleMesh(sp))),
//                sizeType2Dim3(lower),
//                (spParticlePage const *) spParticleDataRoot(sp),
//                page_id_d,
//                disp_d);
//
//
//    spParallelHostAlloc(data_displs, sizeof(size_type) * (*num_of_page));
//    spParallelMemcpy(*data_displs, disp_d, sizeof(size_type) * (*num_of_page));
//
//
//    spParallelHostAlloc(page_id, sizeof(MeshEntityId) * (*num_of_page));
//    spParallelMemcpy(*page_id, page_id_d, sizeof(MeshEntityId) * (*num_of_page));
//
//    spParallelDeviceFree(&disp_d);
//    spParallelDeviceFree(&page_id_d);
//    spParallelDeviceFree(&page_offset_d);
//    spParallelHostFree(&page_offset_h);
//
//    return SP_SUCCESS;
//};
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

