//
// Created by salmon on 16-6-14.
//
//

extern "C" {
#include <assert.h>
#include <math.h>
#include "../sp_lite_def.h"
#include "../spParallel.h"
#include "../spMesh.h"
#include "../spParticle.h"
#include "../spField.h"
#include "../spPICBoris.h"
#include "spParallelCUDA.h"
#include "../spRandom.h"
}
//



#include</usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>


__global__ void
spBorisInitializeParticleKernel(void **data)
{

    boris_particle *d = (boris_particle *) ((byte_type *) (data) + blockIdx.x +
                                            (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x);
    int s = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);

    d->id[s] = 0;
    d->rx[s] = 0.15;
    d->ry[s] = 0.25;
    d->rz[s] = 0.35;
    d->vx[s] = 1;
    d->vy[s] = 2;
    d->vz[s] = 3;

}



/******************************************************************************************/


__global__ void spBorisYeeUpdateParticleKernel(void *data,
                                               const Real *Ex,
                                               const Real *Ey,
                                               const Real *Ez,
                                               const Real *Bx,
                                               const Real *By,
                                               const Real *Bz,
                                               Real3 inv_dv,
                                               Real cmr_dt,
                                               int3 offset)
{
    boris_particle
            *d = (boris_particle *) ((byte_type *) (data) + blockIdx.x +
                                     (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x);
    boris_particle *s = (boris_particle *) ((byte_type *) (data) +
                                            (blockIdx.x + offset.x + gridDim.x) % gridDim.x +
                                            (blockIdx.y + offset.y + gridDim.y) % gridDim.y * gridDim.x +
                                            (blockIdx.z + offset.z + gridDim.z) % gridDim.z * gridDim.y * gridDim.x);

    int s_tail = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
    __shared__ int d_tail;
    __syncthreads();
    if (s_tail == 0) { d_tail = 0; }
    __syncthreads();
    MeshEntityShortId old_id;
    old_id.v = s->id[s_tail];
    if (old_id.x == offset.x && old_id.y == offset.y && old_id.z == offset.z)
    {
        Real rx = s->rx[s_tail];
        Real ry = s->ry[s_tail];
        Real rz = s->rz[s_tail];
        Real vx = s->vx[s_tail];
        Real vy = s->vy[s_tail];
        Real vz = s->vz[s_tail];

        s->id[s_tail] = 0xFF; // TODO this should be atomic

        spBoris(cmr_dt, inv_dv, Ex, Ey, Ez, Bx, By, Bz, &rx, &ry, &rz, &vx, &vy, &vz);

        MeshEntityShortId id;

        id.x = (int8_t) (floor(rx));
        rx -= (Real) (id.x);
        id.y = (int8_t) (floor(ry));
        ry -= (Real) (id.y);
        id.z = (int8_t) (floor(rz));
        rz -= (Real) (id.z);

        id.x += old_id.x;
        id.y += old_id.y;
        id.z += old_id.z;
        int s = 0;
        while (1)
        {
            s = atomicAdd(&d_tail, 1);
            if (d->id[s] != 0xFF) break;
        }


        d->id[s] = id.v;
        d->rx[s] = rx;
        d->ry[s] = ry;
        d->rz[s] = rz;
        d->vx[s] = vx;
        d->vy[s] = vy;
        d->vz[s] = vz;
    }

};

int spBorisYeeParticleUpdate(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{

    Real3 inv_dv;

    Real const *dx = spMeshGetDx(spMeshAttrMesh((spMeshAttr const *) sp));

    inv_dv.x = dt / dx[0];
    inv_dv.y = dt / dx[1];
    inv_dv.z = dt / dx[2];

    Real cmr_dt = dt * spParticleGetCharge(sp) / spParticleGetMass(sp);

    dim3 dims = sizeType2Dim3(spMeshGetDims(spMeshAttrMesh((spMeshAttr const *) sp)));



//    int3 global_start;
//    for (global_start.x = -1; global_start.x <= 1; ++global_start.x)
//        for (global_start.y = -1; global_start.y <= 1; ++global_start.y)
//            for (global_start.z = -1; global_start.z <= 1; ++global_start.z)
//            {
//                CALL_KERNEL(spBorisYeeUpdateParticleKernel,
//                            sizeType2Dim3(spMeshArrayShape(spParticleMesh(sp))),
//                            spParticleFiberLength(sp),
//                            spParticleDeviceData(sp),
//                            (Real const *) spFieldDeviceDataConst(fE),
//                            (Real const *) spFieldDeviceDataConst(fB),
//                            inv_dv, cmr_dt, global_start);
//            }

//    CALL_KERNEL(spUpdateParticleBorisScatterBlockKernel,
//                sizeType2Dim3(spMeshArrayShape(spParticleMesh(sp))),
//                spParticleFiberLength(sp),
//                spParticleDeviceData(sp),
//                (fRho->device_data), (fJ->device_data));

//    spParallelDeviceSync();        //wait for iteration to finish

    SP_CALL(spParticleSync(sp));
    SP_CALL(spFieldSync(fJ));
    SP_CALL(spFieldSync(fRho));
    return SP_SUCCESS;
}




//__global__ void
//spBorisInitializeParticleKernel(boris_particle *d, spParticleFiber **bucket, size_type PIC)
//{
//
////    size_type block_num = spParallelBlockNum();
////
////    while (PIC > 0)
////    {
////        __syncthreads();
////
////        if ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) == 0)
////        {
////            spParticlePage *t = (spParticlePage *) spPageAtomicPop((spPage **) pool);
////            t->next = bucket[block_num];
////            bucket[block_num] = t;
////        }
////        __syncthreads();
////
////        int s = bucket[block_num]->offset
////            + (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
////        if (s < PIC)
////        {
////            d->id[s].v = 0;
////            d->rx[s] = 0.15;
////            d->ry[s] = 0.25;
////            d->rz[s] = 0.35;
////            d->vx[s] = 1;
////            d->vy[s] = 2;
////            d->vz[s] = 3;
////        }
////        PIC -= SP_NUMBER_OF_ENTITIES_IN_PAGE;
////    }
//
//}
//
//
///******************************************************************************************/
//
//__constant__ Real cmr_dt;
//
//__constant__ int3 mesh_offset;
//
//__constant__ int SP_MESH_NUM_OF_ENTITY_IN_GRID;
//
//__constant__ float3 mesh_inv_dv;
//
//#define ll 0
//#define rr 1
//#define RADIUS 2
//#define CACHE_EXTENT_X RADIUS*2
//#define CACHE_EXTENT_Y RADIUS*2
//#define CACHE_EXTENT_Z RADIUS*2
//#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)
//#define IX  1
//#define IY  CACHE_EXTENT_X
//#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y
//
//__device__ void cache_gather(Real *v, Real const *f, Real rx, Real ry, Real rz)
//{
//
//    *v = *v
//         + (f[IX + IY + IZ /*  */] * (rx - ll) * (ry - ll) * (rz - ll)
//            + f[IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
//            + f[IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
//            + f[IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
//            + f[IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
//            + f[IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
//            + f[IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
//            + f[0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz));
//}
//
//#undef ll
//#undef rr
//#undef IX
//#undef IY
//#undef IZ
//
//__global__ void spBorisYeeUpdateParticleKernel(boris_particle *d,
//                                               boris_particle *bucket,
//                                               const Real *tE,
//                                               const Real *tB,
//                                               float3 inv_dv,
//                                               Real cmr_dt)
//{
//
//    spParticlePage *dest, *src;
//    __shared__ int g_d_tail, g_s_tail;
//    MeshEntityId tag;
//
//    __syncthreads();
//
//    if ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) == 0)
//    {
//        g_s_tail = 0;
//        g_d_tail = 0;
//    }
//    __syncthreads();
//
//
//    tag.x = (int16_t) (blockIdx.x);
//    tag.y = (int16_t) (blockIdx.y);
//    tag.z = (int16_t) (blockIdx.z);
//    src = bucket[blockIdx.x + (blockIdx.y * gridDim.z + blockIdx.z) * gridDim.z];
//    dest = bucket[blockIdx.x + (blockIdx.y * gridDim.z + blockIdx.z) * gridDim.z];
//
//    for (int d_tail = 0, s_tail = 0;
////         spParticleMapAndPack(nullptr, &dest, &src, &d_tail, &g_d_tail, &s_tail, &g_s_tail, pool, tag)!= SP_MP_FINISHED
//            ;)
//    {
//
//        MeshEntityId old_id = d->flag[s_tail];
//        Real rx = d->rx[s_tail];
//        Real ry = d->ry[s_tail];
//        Real rz = d->rz[s_tail];
//        Real vx = d->vx[s_tail];
//        Real vy = d->vy[s_tail];
//        Real vz = d->vz[s_tail];
//
//        Real ax, ay, az;
//        Real tx, ty, tz;
//
//        Real tt;
//
//        cache_gather(&ax, tE + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
//        cache_gather(&ay, tE + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
//        cache_gather(&az, tE + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
//
//        cache_gather(&tx, tB + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
//        cache_gather(&ty, tB + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
//        cache_gather(&tz, tB + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);
//
//        ax *= cmr_dt;
//        ay *= cmr_dt;
//        az *= cmr_dt;
//
//        tx *= cmr_dt;
//        ty *= cmr_dt;
//        tz *= cmr_dt;
//
//        rx += vx * 0.5 * inv_dv.x;
//        ry += vy * 0.5 * inv_dv.y;
//        rz += vz * 0.5 * inv_dv.z;
//
//        vx += ax;
//        vy += ay;
//        vz += az;
//
//        Real v_x, v_y, v_z;
//
//        v_x = vx + (vy * tz - vz * ty);
//        v_y = vy + (vz * tx - vx * tz);
//        v_z = vz + (vx * ty - vy * tx);
//
//        tt = 2 / (tx * tx + ty * ty + tz * tz + 1);
//
//        vx += ax + (v_y * tz - v_z * ty) * tt;
//        vy += ax + (v_z * tx - v_x * tz) * tt;
//        vz += ax + (v_x * ty - v_y * tx) * tt;
//
//        rx += vx * 0.5 * mesh_inv_dv.x;
//        ry += vy * 0.5 * mesh_inv_dv.y;
//        rz += vz * 0.5 * mesh_inv_dv.z;
//        MeshEntityId id;
//
//        /*    @formatter:off */
//		id.x = (int16_t) (floor(rx));rx -= (Real) (id.x);
//		id.y = (int16_t) (floor(ry));ry -= (Real) (id.y);
//		id.z = (int16_t) (floor(rz));rz -= (Real) (id.z);
//
//		id.x += old_id.x;
//		id.y += old_id.y;
//		id.z += old_id.z;
//		/*    @formatter:on */
//        d->flag[d_tail] = id;
//        d->rx[d_tail] = rx;
//        d->ry[d_tail] = ry;
//        d->rz[d_tail] = rz;
//        d->vx[d_tail] = vx;
//        d->vy[d_tail] = vy;
//        d->vz[d_tail] = vz;
//    }
//
//};
//void spBorisYeeParticleUpdate(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
//{
//
//    Real3 inv_dv;
//    inv_dv.x = dt / sp->m->dx.x;
//    inv_dv.y = dt / sp->m->dx.y;
//    inv_dv.z = dt / sp->m->dx.z;
//
//    Real cmr_dt = dt * sp->charge / sp->mass;
//
//    spBorisYeeUpdateParticleKernel << < sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >> > (sp->m_buckets_, sp->m_page_pool_,
//        (Real const *) fE->device_data, (Real const *) fB->device_data, inv_dv, cmr_dt);
//
//    /*    @formatter:off */
//
////	spUpdateParticleBorisScatterBlockKernel<<< sp->m->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
////			(fRho->device_data), ( fJ->device_data));
//	/*    @formatter:on */
//
//    spParallelDeviceSync();        //wait for iteration to finish
//
//    spParticleSync(sp);
//    spFieldSync(fJ);
//    spFieldSync(fRho);
//
//}
//__global__ void spUpdateParticleBorisPushBlockKernel(spParticlePage **buckets, const Real *fE, const Real *fB, Real3 inv_dv,
//		Real cmr_dt)
//{
//
//
//	__shared__ Real tE[24], tB[3 * 8];
//
//
//
//	/*    @formatter:off */
//	MC_FOREACH_BLOCK(THREAD_IDX, BLOCK_DIM, BLOCK_IDX, GRID_DIM)
//	{
//		/*    @formatter:on */
//
//		assert(BLOCK_DIM.x * BLOCK_DIM.y * BLOCK_DIM.z <= NUMBER_OF_THREADS_PER_BLOCK);
//
//		{
//
//			size_type g_f_num = ((BLOCK_IDX.x + THREAD_IDX.x + GRID_DIM.x - RADIUS) % GRID_DIM.x)
//					+ (((BLOCK_IDX.y + THREAD_IDX.y + GRID_DIM.y - RADIUS) % GRID_DIM.y)
//							+ ((BLOCK_IDX.z + THREAD_IDX.z + GRID_DIM.z - RADIUS) % GRID_DIM.z) * GRID_DIM.y)
//							* GRID_DIM.x;
//
//			if (THREAD_IDX.x < 8)
//			{
//
//				tE[0 * 8 + THREAD_IDX.x] = fE[g_f_num * 3 + 0];
//				tE[1 * 8 + THREAD_IDX.x] = fE[g_f_num * 3 + 1];
//				tE[2 * 8 + THREAD_IDX.x] = fE[g_f_num * 3 + 2];
//
//				tB[0 * 8 + THREAD_IDX.x] = fB[g_f_num * 3 + 0];
//				tB[1 * 8 + THREAD_IDX.x] = fB[g_f_num * 3 + 1];
//				tB[2 * 8 + THREAD_IDX.x] = fB[g_f_num * 3 + 2];
//
//			}
//		}
//
//
//
//		spParticlePage *pg = buckets[BLOCK_IDX.x + (BLOCK_IDX.y + BLOCK_IDX.z * GRID_DIM.y) * GRID_DIM.x];
//		while (pg != 0x0)
//		{
//			spUpdateParticleBorisPushThreadKernel(THREAD_IDX.x, BLOCK_DIM.x, pg, tE, tB, inv_dv, cmr_dt);
//			spParallelSyncThreads();
//			pg = pg->next;
//		}
//
//	}
//}
//
//__device__ void spUpdateParticleBorisSortThreadKernel(int THREAD_ID, spParticlePage **dest, spParticlePage const *src, MeshEntityId tag)
//{
//    int s_tail = 0; // return current value and s_tail+=1 equiv. s_tail++
//    int d_tail = 0;
//
//    __shared__ int ps_tail;
//    __shared__ int pd_tail;
//    if (THREAD_ID == 0)
//    {
//        if ((*dest) == NULL)
//        {
//            //NEW PAGE
//        }
//        ps_tail = 0;
//        pd_tail = (*dest)->tail;
//    }
//
//    spParallelThreadSync();
//
//    while (src != NULL)
//    {
//
//        if (THREAD_ID < SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE - d_tail) // guarantee d_tail <SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE
//        {
//            while ((s_tail < SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE))
//            {
//                s_tail = spAtomicInc(&ps_tail, 1);
//                if (P_GET_FLAG(src->data, s_tail).v == tag.v)
//                {
//                    d_tail = spAtomicInc(&pd_tail, 1);
//                    break;
//                }
//            }
//            if (s_tail < SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE)
//            {
//                P_GET_FLAG((*dest)->data, d_tail) = 0;
//                P_GET((*dest)->data, struct boris_s, Real, rx, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, rx, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, ry, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, ry, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, rz, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, rz, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, vx, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, vx, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, vy, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, vy, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, vz, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, vz, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, f, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, f, s_tail);
//                P_GET((*dest)->data, struct boris_s, Real, w, d_tail) =
//                        P_GET(src->data, struct boris_s, Real, w, s_tail);
//                continue;
//            }
//        }
//        spParallelThreadSync();
//        if (d_tail == SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE)
//        {
//            if (THREAD_ID == 0)
//            {
//                (*dest)->tail = SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE;
//
//            }
//
//            dest = &((*dest)->next);
//
//            if (THREAD_ID == 0)
//            {
//
//                if ((*dest) == NULL)
//                {
//                    //NEW PAGE
//                }
//                pd_tail = (*dest)->tail;
//            }
//
//
//        }
//        if (s_tail == SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE)
//        {
//            src = src->next;
//            if (THREAD_ID == 0) { ps_tail = 0; }
//        }
//
//        spParallelThreadSync();
//    }
//}
//
//__global__ void spUpdateParticleBorisSortBlockKernel(spParticlePage **dest_buckets, spParticlePage const **src_buckets)
//{
//    /*    @formatter:off */
//    MC_FOREACH_BLOCK(THREAD_IDX, BLOCK_DIM, BLOCK_IDX, GRID_DIM)
//    {
//   /*    @formatter:on */
//
//
//
//        spParticlePage *dest = dest_buckets[BLOCK_ID];
//
//        assert(dest != 0x0);
//
//        MeshEntityId dest_id;
//        dest_id.x = BLOCK_IDX.x;
//        dest_id.y = BLOCK_IDX.y;
//        dest_id.z = BLOCK_IDX.z;
//        for (int x = -1; x <= 1; ++x)
//            for (int y = -1; y <= 1; ++y)
//                for (int z = -1; z <= 1; ++z)
//                {
//                    spParticlePage const *src = src_buckets[ //
//                            ((BLOCK_IDX.x - x + GRID_DIM.x) % GRID_DIM.x)
//                            + (((BLOCK_IDX.y - y + GRID_DIM.y) % GRID_DIM.y) +
//                               ((BLOCK_IDX.z - z + GRID_DIM.z) % GRID_DIM.z) * GRID_DIM.y) * GRID_DIM.x];
//
//
//                    while (src != 0x0)
//                    {
//                        spUpdateParticleBorisSortThreadKernel(THREAD_IDX.x, BLOCK_DIM.x, dest, src, dest_id);
//                       __syncthreads();
//
//                        src = (src->next);
//                    }
//
//                }
//    }
//}
//
//__global__ void spUpdateParticleBorisScatterBlockKernel(spParticlePage **buckets, Real *fRho, Real *fJ)
//{
//    float4 J4;
//    J4.x = 0;
//    J4.y = 0;
//    J4.z = 0;
//    J4.w = 0;
//    /*    @formatter:off */
//    MC_FOREACH_BLOCK(THREAD_IDX, BLOCK_DIM, BLOCK_IDX, GRID_DIM)
//    {
//   /*    @formatter:on */
//        spParticlePage const *pg = buckets[BLOCK_ID];
//
//        while (pg != 0x0)
//        {
//
//            // FIXME THIS IS WRONG!!!
//
//
//            for (size_type s = THREAD_IDX.x; s < SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE; s += BLOCK_DIM.x)
//            {
//
//                if (P_GET(pg->data, struct boris_s, MeshEntityId, id, s).v == 0x0)
//                {
//                    Real rx = P_GET(pg->data, struct boris_s, Real, rx, s);
//                    Real ry = P_GET(pg->data, struct boris_s, Real, ry, s);
//                    Real rz = P_GET(pg->data, struct boris_s, Real, rz, s);
//                    Real vx = P_GET(pg->data, struct boris_s, Real, vx, s);
//                    Real vy = P_GET(pg->data, struct boris_s, Real, vy, s);
//                    Real vz = P_GET(pg->data, struct boris_s, Real, vz, s);
//                    Real f = P_GET(pg->data, struct boris_s, Real, f, s);
//                    Real w = P_GET(pg->data, struct boris_s, Real, w, s);
//
//                    Real w0 = abs((rx - 0.5) * (ry - 0.5) * (rz - 0.5)) * f * w;
//
//                    J4.w += w0;
//                    J4.x += w0 * vx;
//                    J4.y += w0 * vy;
//                    J4.z += w0 * vz;
//                }
//
//            }
//
//            pg = pg->next;
//        }
//
//        SP_ATOMIC_ADD(&(fJ[BLOCK_ID + SP_MESH_NUM_OF_ENTITY_IN_GRID * 0]), J4.x);
//        SP_ATOMIC_ADD(&(fJ[BLOCK_ID + SP_MESH_NUM_OF_ENTITY_IN_GRID * 1]), J4.y);
//        SP_ATOMIC_ADD(&(fJ[BLOCK_ID + SP_MESH_NUM_OF_ENTITY_IN_GRID * 2]), J4.z);
//        SP_ATOMIC_ADD(&(fRho[BLOCK_ID]), J4.w);
//    }
//
//
//}
//

//
///***************************************************************************************************************/
//
//__global__ void spUpdateField_Yee_kernel(const Real *fJ, Real *fE, Real *fB)
//{
//}
//
//void spUpdateFieldFDTD(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
//{
//
//    /*    @formatter:off */
//
//    spUpdateField_Yee_kernel<<< ctx->topology_dims, NUMBER_OF_THREADS_PER_BLOCK >>> (((Real *) fJ->device_data),
//            ((Real *) fE->device_data), ((Real *) fB->device_data));
//    /*    @formatter:on */
//
//    spParallelThreadSync();        //wait for iteration to finish
//    spFieldSync(fE);
//    spFieldSync(fB);
//}
//
