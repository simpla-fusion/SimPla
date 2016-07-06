//
// Created by salmon on 16-6-14.
//

#include "Boris.h"
#include <stdio.h>
#include <assert.h>

#include "sp_lite_def.h"
#include "spField.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spPage.h"

/******************************************************************************************/

void spBorisYeeInitializeParticle(spParticle *sp, size_type NUM_OF_PIC)
{
	assert(sp != 0x0);

	spParticleAddAttribute(sp, "rx", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "ry", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "rz", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "vx", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "vy", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "vz", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "f", SP_TYPE_Real, sizeof(Real));
	spParticleAddAttribute(sp, "w", SP_TYPE_Real, sizeof(Real));

	spParticleDeploy(sp, NUM_OF_PIC);

	spParticleSync(sp);

}

/******************************************************************************************/

__constant__ Real cmr_dt;
__constant__ int3 mesh_offset;
__constant__ int SP_MESH_NUM_OF_ENTITY_IN_GRID;
__constant__ float3 mesh_inv_dv;

#define ll 0
#define rr 1.0
#define RADIUS 2
#define CACHE_EXTENT_X RADIUS*2
#define CACHE_EXTENT_Y RADIUS*2
#define CACHE_EXTENT_Z RADIUS*2
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)
#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y
__device__
void cache_gather(Real *v, Real const *f, Real rx, Real ry, Real rz)
{

	*v = *v
			+ (f[ IX + IY + IZ /*    */] * (rx - ll) * (ry - ll) * (rz - ll)
					+ f[ IX + IY /*     */] * (rx - ll) * (ry - ll) * (rr - rz)
					+ f[ IX + IZ /*     */] * (rx - ll) * (rr - ry) * (rz - ll)
					+ f[ IX /*          */] * (rx - ll) * (rr - ry) * (rr - rz)
					+ f[ IY + IZ /*     */] * (rr - rx) * (ry - ll) * (rz - ll)
					+ f[ IY /*          */] * (rr - rx) * (ry - ll) * (rr - rz)
					+ f[ IZ /*          */] * (rr - rx) * (rr - ry) * (rz - ll)
					+ f[0 /*            */] * (rr - rx) * (rr - ry) * (rr - rz)) * cmr_dt;
}

#undef ll
#undef rr
#undef IX
#undef IY
#undef IZ

__global__ void spUpdateParticle_push_Boris_Kernel(spPage** buckets, const Real *fE, const Real *fB)
{

	__shared__ Real tE[24], tB[3 * 8];

	__shared__ Real __align__(8) _ax[NUMBER_OF_THREADS_PER_BLOCK];
	__shared__ Real __align__(8) _ay[NUMBER_OF_THREADS_PER_BLOCK];
	__shared__ Real __align__(8) _az[NUMBER_OF_THREADS_PER_BLOCK];

	__shared__ Real __align__(8) _vx[NUMBER_OF_THREADS_PER_BLOCK];
	__shared__ Real __align__(8) _vy[NUMBER_OF_THREADS_PER_BLOCK];
	__shared__ Real __align__(8) _vz[NUMBER_OF_THREADS_PER_BLOCK];

	__shared__ Real __align__(8) _tx[NUMBER_OF_THREADS_PER_BLOCK];
	__shared__ Real __align__(8) _ty[NUMBER_OF_THREADS_PER_BLOCK];
	__shared__ Real __align__(8) _tz[NUMBER_OF_THREADS_PER_BLOCK];

	__shared__ Real __align__(8) _tt[NUMBER_OF_THREADS_PER_BLOCK];

	assert(blockDim.x * blockDim.y * blockDim.z<=NUMBER_OF_THREADS_PER_BLOCK);

	{

		int g_f_num = ((blockIdx.x + threadIdx.x + gridDim.x - RADIUS) % gridDim.x)
				+ (((blockIdx.y + threadIdx.y + gridDim.y - RADIUS) % gridDim.y)
						+ ((blockIdx.z + threadIdx.z + gridDim.z - RADIUS) % gridDim.z) * gridDim.y) * gridDim.x;

		if (threadIdx.x < 8)
		{

			tE[0 * 8 + threadIdx.x] = fE[g_f_num * 3 + 0];
			tE[1 * 8 + threadIdx.x] = fE[g_f_num * 3 + 1];
			tE[2 * 8 + threadIdx.x] = fE[g_f_num * 3 + 2];

			tB[0 * 8 + threadIdx.x] = fB[g_f_num * 3 + 0];
			tB[1 * 8 + threadIdx.x] = fB[g_f_num * 3 + 1];
			tB[2 * 8 + threadIdx.x] = fB[g_f_num * 3 + 2];

		}
	}

	__syncthreads();

	struct boris_page_s * pg = (struct boris_page_s *) buckets[blockDim.x
			+ (blockDim.y + blockDim.z * gridDim.y) * gridDim.x];

	while (pg != 0x0)
	{

		for (int s = threadIdx.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += blockDim.x)
		{
#define ax _ax[threadIdx.x]
#define ay _ay[threadIdx.x]
#define az _az[threadIdx.x]

#define tx _tx[threadIdx.x]
#define ty _ty[threadIdx.x]
#define tz _tz[threadIdx.x]

#define vx_ _vx[s]
#define vy_ _vy[s]
#define vz_ _vz[s]

#define tt _tt[threadIdx.x]

			{
				Real rx = pg->rx[s], ry = pg->ry[s], rz = pg->rz[s];

				cache_gather(&ax, tE + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
				cache_gather(&ay, tE + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
				cache_gather(&az, tE + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

				cache_gather(&tx, tB + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
				cache_gather(&ty, tB + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
				cache_gather(&tz, tB + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);
			}

			vx_ = pg->vx[s];
			vy_ = pg->vy[s];
			vz_ = pg->vz[s];

			pg->rx[s] += vx_ * 0.5* mesh_inv_dv.x;
			pg->ry[s] += vy_ * 0.5* mesh_inv_dv.y;
			pg->rz[s] += vz_ * 0.5* mesh_inv_dv.z;

			vx_ += ax;
			vy_ += ay;
			vz_ += az;

			Real v_x, v_y, v_z;
			v_x = vx_ + (vy_ * tz- vz_ * ty);
			v_y = vy_ + (vz_ * tx- vx_ * tz);
			v_z = vz_ + (vx_ * ty- vy_ * tx);

			tt = 2.0 / (tx* tx+ ty * ty + tz * tz + 1.0);

			vx_ += ax+(v_y * tz- v_z * ty) * tt;
			vy_ += ax+(v_z * tx- v_x * tz) * tt;
			vz_ += ax+(v_x * ty- v_y * tx) * tt;

			pg->rx[s] += vx_ * 0.5 * mesh_inv_dv.x;
			pg->ry[s] += vy_ * 0.5 * mesh_inv_dv.y;
			pg->rz[s] += vz_ * 0.5 * mesh_inv_dv.z;

			pg->vx[s] = vx_;
			pg->vy[s] = vy_;
			pg->vz[s] = vz_;

#undef ax
#undef ay
#undef az

#undef tx
#undef ty
#undef tz

#undef vx
#undef vy
#undef vz

#undef tt

		}
		__syncthreads();

		pg =(struct boris_page_s *) ( pg->next);
	}

}

__global__ void spUpdateParticle_sort_Boris_kernel(spPage ** buckets)
{

#define MESH_ID (blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)

	__shared__ bucket_entity_flag_t src_flag;
//	__shared__ bucket_entity_flag_t dest_flag;
	__shared__ unsigned int dest_tail;

	dest_tail = 0;

	struct boris_page_s * dest = (struct boris_page_s *) buckets[MESH_ID];

	assert(dest != 0x0);

	for (int i = 26; i >= 0; --i)
	{

		struct boris_page_s * pg = (struct boris_page_s *) buckets[ //
				((blockIdx.x + SP_NEIGHBOUR_OFFSET[i].x + gridDim.x) % gridDim.x)
						+ (((blockIdx.y + SP_NEIGHBOUR_OFFSET[i].y + gridDim.y) % gridDim.y)
								+ ((blockIdx.z + SP_NEIGHBOUR_OFFSET[i].z + gridDim.z) % gridDim.z) * gridDim.y)
								* gridDim.x];

		assert(pg != 0x0);

		while (pg != 0x0)
		{

			if (threadIdx.x == 0)
			{
				src_flag = pg->flag;
			}

			for (int s = threadIdx.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += blockDim.x)
			{

				if ((src_flag) & 0x3F == SP_NEIGHBOUR_OFFSET_flag[i])
				{

					unsigned int tail = atomicAdd(&dest_tail, 1);

					assert(tail<SP_NUMBER_OF_ENTITIES_IN_PAGE);

					dest->rx[tail] = pg->rx[s];
					dest->ry[tail] = pg->ry[s];
					dest->rz[tail] = pg->rz[s];

					dest->vx[tail] = pg->vx[s];
					dest->vy[tail] = pg->vy[s];
					dest->vz[tail] = pg->vz[s];

					dest->f[tail] = pg->f[s];
					dest->w[tail] = pg->w[s];

				}

			}
			__syncthreads();

			pg = (struct boris_page_s *) (pg->next);
		}

	}
#undef MESH_ID
}

__global__ void spUpdateParticle_scatter_Boris_kernel(spPage ** buckets, Real *fRho, Real *fJ)
{
	float4 J4;
	J4.x = 0;
	J4.y = 0;
	J4.z = 0;
	J4.w = 0;
#define MESH_ID (blockDim.x+mesh_offset.x + (blockDim.y+mesh_offset.y +( blockDim.z+mesh_offset.z) * gridDim.y) * gridDim.x)

	struct boris_page_s const * pg = (struct boris_page_s *) buckets[MESH_ID];

	while (pg != 0x0)
	{

		// FIXME THIS IS WRONG!!!

		bucket_entity_flag_t dest_flag = pg->flag;

		for (int s = threadIdx.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += blockDim.x)
		{
			if (dest_flag & (0x3F << (s * 6)) == 0x15)
			{
				Real w0 = abs((pg->rx[s] - 0.5) * (pg->ry[s] - 0.5) * (pg->rz[s] - 0.5)) * pg->f[s] * pg->w[s];

				J4.w += w0;
				J4.x += w0 * pg->vx[s];
				J4.y += w0 * pg->vy[s];
				J4.z += w0 * pg->vz[s];
			}
			else
			{
				dest_flag = dest_flag & ((0x3F << (s * 6)));
			}
		}

		atomicAnd(&(((struct boris_page_s *) buckets[MESH_ID])->flag), ~(dest_flag));

		pg = (struct boris_page_s *) (pg->next);
	}

	atomicAdd(&(fJ[MESH_ID + SP_MESH_NUM_OF_ENTITY_IN_GRID * 0]), J4.x);
	atomicAdd(&(fJ[MESH_ID + SP_MESH_NUM_OF_ENTITY_IN_GRID * 1]), J4.y);
	atomicAdd(&(fJ[MESH_ID + SP_MESH_NUM_OF_ENTITY_IN_GRID * 2]), J4.z);
	atomicAdd(&(fRho[MESH_ID]), J4.w);

}

void spBorisYeeUpdateParticle(spParticle *pg, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{

	float3 t_inv_dv = make_float3(dt / pg->m->dx.x, dt / pg->m->dx.y, dt / pg->m->dx.z);

	Real t_cmr_dt = 0.5 * dt * pg->charge / pg->mass;

	cudaMemcpyToSymbol(&mesh_inv_dv, &t_inv_dv, sizeof(float3), cudaMemcpyDefault);

	cudaMemcpyToSymbol(&cmr_dt, &t_cmr_dt, sizeof(Real), cudaMemcpyDefault);

	spUpdateParticle_push_Boris_Kernel<<<pg->m->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(pg->buckets,
			((Real*) fE->device_data), ((Real*) fB->device_data));

	spUpdateParticle_sort_Boris_kernel<<<pg->m->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(pg->buckets);

	spUpdateParticle_scatter_Boris_kernel<<<pg->m->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(pg->buckets,
			((Real*) fRho->device_data), ((Real*) fJ->device_data));

	cudaDeviceSynchronize();        //wait for iteration to finish

	spParticleSync(pg);
	spFieldSync(fJ);
	spFieldSync(fRho);

}
/***************************************************************************************************************/

__global__ void spUpdateField_Yee_kernel(const Real *fJ, Real *fE, Real *fB)
{
}
void spUpdateField_Yee(spMesh *ctx, Real dt, const spField *fRho, const spField *fJ, spField *fE, spField *fB)
{
	spUpdateField_Yee_kernel<<<ctx->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(((Real*) fJ->device_data),
			((Real*) fE->device_data), ((Real*) fB->device_data));

	cudaDeviceSynchronize();        //wait for iteration to finish
	spFieldSync( fE);
	spFieldSync( fB);
}

