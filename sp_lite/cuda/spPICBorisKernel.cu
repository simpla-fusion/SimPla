//
// Created by salmon on 16-6-14.
//
//

extern "C" {
#include <assert.h>
#include <math.h>

#include "../sp_lite_def.h"
#include "../spMesh.h"
#include "../spParticle.h"
#include "../spField.h"
#include "../spPICBoris_device.h"
#include "../spRandom.h"
#include "../spPhysicalConstants.h"

#include "spParallelCUDA.h"

}


#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>


__global__ void
spParticleInitializeBorisYeeKernel(boris_particle *sp, dim3 min, dim3 max, dim3 strides, size_type pic, Real vT,
                                   Real f0, int import_sample)
{
    size_type threadId = (threadIdx.x) +
                         (threadIdx.y) * blockDim.x +
                         (threadIdx.z) * blockDim.x * blockDim.y;


    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    for (int x = blockIdx.x + min.x; x < max.x; x += gridDim.x)
        for (int y = blockIdx.y + min.y; y < max.y; y += gridDim.y)
            for (int z = blockIdx.z + min.z; z < max.z; z += gridDim.z)
            {
                size_type s0 = threadId + x * strides.x + y * strides.y + z * strides.z;

                for (size_type s = 0; s < pic; s += num_of_thread)
                {
                    sp->id[s0 + s] = 0;
                    sp->f[s0 + s] = f0;
                    if (import_sample > 0)
                    {
                        sp->f[s0 + s] *= exp(-sp->vx[s0 + s] * sp->vx[s0 + s]
                                             - sp->vy[s0 + s] * sp->vy[s0 + s]
                                             - sp->vz[s0 + s] * sp->vz[s0 + s]
                        );
                    }

                    sp->vx[s0 + s] *= vT;
                    sp->vy[s0 + s] *= vT;
                    sp->vz[s0 + s] *= vT;

                    sp->w[s0 + s] = 0;
                }
            }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0, int do_import_sample)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    SP_CALL(spParticleDeploy(sp));

    size_type max_number_of_entities = spParticleGetNumberOfEntities(sp);

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];
    SP_CALL(spMeshGetDx(m, dx));
    Real vT = sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));
    Real f0 = n0 * dx[0] * dx[1] * dx[2] / spParticleGetPIC(sp);

    size_type x_min[3], x_max[3], strides[3];
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_CENTER, x_min, x_max, strides));
    strides[0] *= spParticleGetMaxPIC(sp);
    strides[1] *= spParticleGetMaxPIC(sp);
    strides[2] *= spParticleGetMaxPIC(sp);

    size_type blocks[3] = {16, 1, 1};
    size_type threads[3]{SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE, 1, 1};

    void **device_data;

    spParticleGetAllAttributeData_device(sp, &device_data);

    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel,
                          sizeType2Dim3(blocks), sizeType2Dim3(threads),
                          (boris_particle *) device_data,
                          sizeType2Dim3(x_min), sizeType2Dim3(x_max), sizeType2Dim3(strides),
                          spParticleGetPIC(sp), vT, f0, do_import_sample
    );

    spParallelDeviceFree((void **) &device_data);
    return SP_SUCCESS;
}

/******************************************************************************************/
#define SP_MAX_NUM_OF_PARTICLE_ATTR 32
typedef struct
{
    Real3 inv_dv;

    Real cmr_dt;

    size_type max_pic;

    void *data[SP_MAX_NUM_OF_PARTICLE_ATTR];

    Real *rho;
    Real *J[3];
    const Real *E[3];
    const Real *B[3];


} boris_update_param;

__global__ void
spBorisYeeUpdateParticleKernel(boris_update_param *para, dim3 min, dim3 max, dim3 strides)
{
    size_type threadId = (threadIdx.x) +
                         (threadIdx.y) * blockDim.x +
                         (threadIdx.z) * blockDim.x * blockDim.y;


    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    __shared__ Real Ex[27], Ey[27], Ez[27], Bx[27], By[27], Bz[27], Jx[27], Jy[27], Jz[27], rho[27];

    size_type IX = 1, IY = 3, IZ = 9;
    size_type s_c = 13;

    boris_particle *sp = (boris_particle *) para->data;

    for (int x = blockIdx.x + min.x; x < max.x; x += gridDim.x)
        for (int y = blockIdx.y + min.y; y < max.y; y += gridDim.y)
            for (int z = blockIdx.z + min.z; z < max.z; z += gridDim.z)
            {

                size_type s0 = threadId + x * strides.x + y * strides.y + z * strides.z;

                for (size_type s = 0; s < para->max_pic; s += num_of_thread)
                {
                    Real rx = sp->rx[s0 + s];
                    Real ry = sp->ry[s0 + s];
                    Real rz = sp->rz[s0 + s];
                    Real vx = sp->vx[s0 + s];
                    Real vy = sp->vy[s0 + s];
                    Real vz = sp->vz[s0 + s];
                    Real f = sp->f[s0 + s];
                    Real w = sp->w[s0 + s];
                    sp->id[s0 + s] = 0xFF; // TODO this should be atomic

                    spBoris(para->cmr_dt, para->inv_dv, s_c, IX, IY, IZ,
                            rho, Jx, Jy, Jz,
                            Ex, Ey, Ez, Bx, By, Bz, &rx, &ry, &rz, &vx, &vy, &vz, &f, &w);


                    sp->id[s0 + s] = 0;
                    sp->rx[s0 + s] = rx;
                    sp->ry[s0 + s] = ry;
                    sp->rz[s0 + s] = rz;
                    sp->vx[s0 + s] = vx;
                    sp->vy[s0 + s] = vy;
                    sp->vz[s0 + s] = vz;
                    sp->f[s0 + s] = f;
                    sp->w[s0 + s] = w;
                }
            }
};

int spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    Real cmr_dt = dt * spParticleGetCharge(sp) / spParticleGetMass(sp);

    Real inv_dv[3];
    Real dx[3];
    size_type dims[3];

    SP_CALL(spMeshGetDx(spMeshAttributeGetMesh((spMeshAttribute const *) sp), dx));
    SP_CALL(spMeshGetInvDx(spMeshAttributeGetMesh((spMeshAttribute const *) sp), inv_dv));
    SP_CALL(spMeshGetDims(spMeshAttributeGetMesh((spMeshAttribute const *) sp), dims));


    boris_update_param update_param;


    SP_CALL(spParticleGetAllAttributeData(sp, update_param.data));
    SP_CALL(spFieldSubArray(fRho, (void **) &update_param.rho));
    SP_CALL(spFieldSubArray(fJ, (void **) update_param.J));
    SP_CALL(spFieldSubArray((spField *) fE, (void **) update_param.E));
    SP_CALL(spFieldSubArray((spField *) fB, (void **) update_param.B));

    boris_update_param *device_update_param;

    spParallelDeviceAlloc((void **) &device_update_param, sizeof(boris_update_param));
    spParallelMemcpy((void *) device_update_param, &update_param, sizeof(boris_update_param));

    size_type x_min[3], x_max[3], strides[3];
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_CENTER, x_min, x_max, strides));
    size_type blocks[3] = {16, 1, 1};
    size_type threads[3]{SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE, 1, 1};
    SP_DEVICE_CALL_KERNEL(spBorisYeeUpdateParticleKernel,
                          sizeType2Dim3(blocks), sizeType2Dim3(threads),
                          device_update_param,
                          sizeType2Dim3(x_min), sizeType2Dim3(x_max), sizeType2Dim3(strides)
    );
    spParallelDeviceFree((void **) &device_update_param);

    SP_CALL(spParticleSync(sp));
    SP_CALL(spFieldSync(fJ));
    SP_CALL(spFieldSync(fRho));
    return SP_SUCCESS;
}



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
//void spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
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
//void spFDTDUpdate(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
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
