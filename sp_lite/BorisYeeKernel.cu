//
// Created by salmon on 16-6-14.
//

#include <assert.h>
#include <cmath>
#include "sp_lite_def.h"
#include "spField.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spPage.h"
#include "BorisYee.h"

/******************************************************************************************/

MC_CONSTANT Real cmr_dt;
MC_CONSTANT int3 mesh_offset;
MC_CONSTANT int SP_MESH_NUM_OF_ENTITY_IN_GRID;
MC_CONSTANT float3 mesh_inv_dv;

#define ll 0
#define rr 1
#define RADIUS 2
#define CACHE_EXTENT_X RADIUS*2
#define CACHE_EXTENT_Y RADIUS*2
#define CACHE_EXTENT_Z RADIUS*2
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)
#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y

MC_DEVICE void cache_gather(Real *v, Real const *f, Real rx, Real ry, Real rz)
{

	*v = *v
			+ (f[IX + IY + IZ /*  */] * (rx - ll) * (ry - ll) * (rz - ll)
					+ f[IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
					+ f[IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
					+ f[IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
					+ f[IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
					+ f[IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
					+ f[IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
					+ f[0 /*           */] * (rr - rx) * (rr - ry) * (rr - rz));
}

#undef ll
#undef rr
#undef IX
#undef IY
#undef IZ

MC_GLOBAL void spUpdateParticleBorisPushKernel(spPage **pg, const Real *tE, const Real *tB, Real3 inv_dv, Real cmr_dt)
{

	spPage *dest, *src;
	MC_SHARED int g_d_tail, g_s_tail;
	MeshEntityId tag;

	spParallelSyncThreads();

	if (spParallelThreadNum() == 0)
	{
		g_s_tail = 0;
		g_d_tail = 0;
	}
	spParallelSyncThreads();

	dim3 block_idx = spParallelBlockIdx();
	dim3 grid_dims = spParallelGridDims();

	tag.x = (int16_t) (block_idx.x);
	tag.y = (int16_t) (block_idx.y);
	tag.z = (int16_t) (block_idx.z);
	src = pg[block_idx.x + (block_idx.y * grid_dims.z + block_idx.z) * grid_dims.z];
	dest = pg[block_idx.x + (block_idx.y * grid_dims.z + block_idx.z) * grid_dims.z];

	for (int d_tail = 0, s_tail = 0;
			spParticleMapAndPack(&dest, (const spPage **) &src, &d_tail, &g_d_tail, &s_tail, &g_s_tail, tag)
					!= SP_MP_FINISHED;)
	{

		MeshEntityId old_id = P_GET_FLAG(src->data, s_tail);
		Real rx = P_GET((src)->data, struct boris_s, Real, rx, s_tail);
		Real ry = P_GET((src)->data, struct boris_s, Real, ry, s_tail);
		Real rz = P_GET((src)->data, struct boris_s, Real, rz, s_tail);
		Real vx = P_GET((src)->data, struct boris_s, Real, vx, s_tail);
		Real vy = P_GET((src)->data, struct boris_s, Real, vy, s_tail);
		Real vz = P_GET((src)->data, struct boris_s, Real, vz, s_tail);

		Real ax, ay, az;
		Real tx, ty, tz;

		Real tt;

		cache_gather(&ax, tE + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
		cache_gather(&ay, tE + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
		cache_gather(&az, tE + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

		cache_gather(&tx, tB + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
		cache_gather(&ty, tB + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
		cache_gather(&tz, tB + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

		ax *= cmr_dt;
		ay *= cmr_dt;
		az *= cmr_dt;

		tx *= cmr_dt;
		ty *= cmr_dt;
		tz *= cmr_dt;

		rx += vx * 0.5 * inv_dv.x;
		ry += vy * 0.5 * inv_dv.y;
		rz += vz * 0.5 * inv_dv.z;

		vx += ax;
		vy += ay;
		vz += az;

		Real v_x, v_y, v_z;

		v_x = vx + (vy * tz - vz * ty);
		v_y = vy + (vz * tx - vx * tz);
		v_z = vz + (vx * ty - vy * tx);

		tt = 2 / (tx * tx + ty * ty + tz * tz + 1);

		vx += ax + (v_y * tz - v_z * ty) * tt;
		vy += ax + (v_z * tx - v_x * tz) * tt;
		vz += ax + (v_x * ty - v_y * tx) * tt;

		rx += vx * 0.5 * mesh_inv_dv.x;
		ry += vy * 0.5 * mesh_inv_dv.y;
		rz += vz * 0.5 * mesh_inv_dv.z;
		MeshEntityId id;

		/*    @formatter:off */
		id.x = (int16_t) (floor(rx));
		rx -= (Real) (id.x);
		id.y = (int16_t) (floor(ry));
		ry -= (Real) (id.y);
		id.z = (int16_t) (floor(rz));
		rz -= (Real) (id.z);
		id.x += old_id.x;
		id.y += old_id.y;
		id.z += old_id.z;
		/*    @formatter:on */
		P_GET_FLAG(dest->data, d_tail) = id;
		P_GET((dest)->data, struct boris_s, Real, rx, d_tail) = rx;
		P_GET((dest)->data, struct boris_s, Real, ry, d_tail) = ry;
		P_GET((dest)->data, struct boris_s, Real, rz, d_tail) = rz;
		P_GET((dest)->data, struct boris_s, Real, vx, d_tail) = vx;
		P_GET((dest)->data, struct boris_s, Real, vy, d_tail) = vy;
		P_GET((dest)->data, struct boris_s, Real, vz, d_tail) = vz;
	}

}
;

void spBorisYeeUpdateParticle(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{

	float3 inv_dv;
	inv_dv.x = dt / sp->m->dx.x;
	inv_dv.y = dt / sp->m->dx.y;
	inv_dv.z = dt / sp->m->dx.z;

	Real cmr_dt = dt * sp->charge / sp->mass;

	spUpdateParticleBorisPushKernel<<<sp->m->dims,NUMBER_OF_THREADS_PER_BLOCK>>>(sp->buckets,fE->device_data,fB->device_data,inv_dv,cmr_dt);

	/*    @formatter:off */

//	spUpdateParticleBorisScatterBlockKernel<<< sp->m->dims, NUMBER_OF_THREADS_PER_BLOCK >>>(sp->buckets,
//			(fRho->device_data), ( fJ->device_data));
	/*    @formatter:on */

	spParallelDeviceSync();        //wait for iteration to finish

	spParticleSync(sp);
	spFieldSync(fJ);
	spFieldSync(fRho);

}
//MC_GLOBAL void spUpdateParticleBorisPushBlockKernel(spPage **buckets, const Real *fE, const Real *fB, Real3 inv_dv,
//		Real cmr_dt)
//{
//
//
//	MC_SHARED Real tE[24], tB[3 * 8];
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
//		spPage *pg = buckets[BLOCK_IDX.x + (BLOCK_IDX.y + BLOCK_IDX.z * GRID_DIM.y) * GRID_DIM.x];
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
//MC_DEVICE void spUpdateParticleBorisSortThreadKernel(int THREAD_ID, spPage **dest, spPage const *src, MeshEntityId tag)
//{
//    int s_tail = 0; // return current value and s_tail+=1 equiv. s_tail++
//    int d_tail = 0;
//
//    MC_SHARED int ps_tail;
//    MC_SHARED int pd_tail;
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
//        if (THREAD_ID < SP_NUMBER_OF_ENTITIES_IN_PAGE - d_tail) // guarantee d_tail <SP_NUMBER_OF_ENTITIES_IN_PAGE
//        {
//            while ((s_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE))
//            {
//                s_tail = spAtomicInc(&ps_tail, 1);
//                if (P_GET_FLAG(src->data, s_tail).v == tag.v)
//                {
//                    d_tail = spAtomicInc(&pd_tail, 1);
//                    break;
//                }
//            }
//            if (s_tail < SP_NUMBER_OF_ENTITIES_IN_PAGE)
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
//        if (d_tail == SP_NUMBER_OF_ENTITIES_IN_PAGE)
//        {
//            if (THREAD_ID == 0)
//            {
//                (*dest)->tail = SP_NUMBER_OF_ENTITIES_IN_PAGE;
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
//        if (s_tail == SP_NUMBER_OF_ENTITIES_IN_PAGE)
//        {
//            src = src->next;
//            if (THREAD_ID == 0) { ps_tail = 0; }
//        }
//
//        spParallelThreadSync();
//    }
//}
//
//MC_GLOBAL void spUpdateParticleBorisSortBlockKernel(spPage **dest_buckets, spPage const **src_buckets)
//{
//    /*    @formatter:off */
//    MC_FOREACH_BLOCK(THREAD_IDX, BLOCK_DIM, BLOCK_IDX, GRID_DIM)
//    {
//   /*    @formatter:on */
//
//
//
//        spPage *dest = dest_buckets[BLOCK_ID];
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
//                    spPage const *src = src_buckets[ //
//                            ((BLOCK_IDX.x - x + GRID_DIM.x) % GRID_DIM.x)
//                            + (((BLOCK_IDX.y - y + GRID_DIM.y) % GRID_DIM.y) +
//                               ((BLOCK_IDX.z - z + GRID_DIM.z) % GRID_DIM.z) * GRID_DIM.y) * GRID_DIM.x];
//
//
//                    while (src != 0x0)
//                    {
//                        spUpdateParticleBorisSortThreadKernel(THREAD_IDX.x, BLOCK_DIM.x, dest, src, dest_id);
//                        spParallelSyncThreads();
//
//                        src = (src->next);
//                    }
//
//                }
//    }
//}
//
//MC_GLOBAL void spUpdateParticleBorisScatterBlockKernel(spPage **buckets, Real *fRho, Real *fJ)
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
//        spPage const *pg = buckets[BLOCK_ID];
//
//        while (pg != 0x0)
//        {
//
//            // FIXME THIS IS WRONG!!!
//
//
//            for (size_type s = THREAD_IDX.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += BLOCK_DIM.x)
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
//MC_GLOBAL void spUpdateField_Yee_kernel(const Real *fJ, Real *fE, Real *fB)
//{
//}
//
//void spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
//{
//
//    /*    @formatter:off */
//
//    spUpdateField_Yee_kernel<<< ctx->dims, NUMBER_OF_THREADS_PER_BLOCK >>> (((Real *) fJ->device_data),
//            ((Real *) fE->device_data), ((Real *) fB->device_data));
//    /*    @formatter:on */
//
//    spParallelThreadSync();        //wait for iteration to finish
//    spFieldSync(fE);
//    spFieldSync(fB);
//}
//
