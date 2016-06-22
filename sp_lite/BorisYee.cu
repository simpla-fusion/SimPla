//
// Created by salmon on 16-6-14.
//

#include "Boris.h"
#include <stdio.h>
#include <assert.h>

#include "sp_def.h"
#include "spField.h"
#include "spMesh.h"
#include "spParticle.h"
#include "spPage.h"

//#include "spMesh.cu"
//#include "spField.cu"
//#include "spParticle.cu"
#ifndef NUMBER_OF_THREADS_PER_BLOCK
#	define NUMBER_OF_THREADS_PER_BLOCK 128
#endif //NUMBER_OF_THREADS_PER_BLOCK
/******************************************************************************************/

__global__ void spInitializeParticle_BorisYee_Kernel(spPage** buckets)
{
//
//	index_type g_x = (blockIdx.x);
//	index_type g_y = (blockIdx.y);
//	index_type g_z = (blockIdx.z);
//
//	size_type g_dim_x = gridDim.x;
//	size_type g_dim_y = gridDim.y;
//	size_type g_dim_z = gridDim.z;
//
//	index_type t_x = (threadIdx.x);
//	index_type t_y = (threadIdx.y);
//	index_type t_z = (threadIdx.z);
//	size_type t_dim_x = blockDim.x;
//	size_type t_dim_y = blockDim.y;
//	size_type t_dim_z = blockDim.z;
//
//	size_type g_num = g_x + (g_y + g_z * g_dim_y) * g_dim_x;
//	size_type t_num = t_x + (t_y + t_z * t_dim_y) * t_dim_x;
//
//	boris_point_s*p = (boris_point_s*) (buckets[g_num]->data + (t_num * entity_size_in_byte));
//
//	p->r[0] = 0.5;
//	p->r[1] = 0.5;
//	p->r[2] = 0.5;
//
//	p->v[0] = 0.5;
//	p->v[1] = 0.5;
//	p->v[2] = 0.5;
//
//	p->f = 1.0;
//	p->w = 1.0;

}

void spInitializeParticle_BorisYee(spMesh *ctx, sp_particle_type *sp, size_type NUM_OF_PIC)
{

//	cudaStream_t s_shared[ctx->number_of_shared_blocks];
//
//	for (int i = 0, ie = ctx->number_of_shared_blocks; i < ie; ++i)
//	{
//		cudaStreamCreate(&s_shared[i]);
//		spInitializeParticle_BorisYee_Kernel<<<ctx->shared_blocks[i], ctx->threadsPerBlock, 0, s_shared[i]>>>(
//				(spMesh *) spObject_device_((spObject*) ctx), (sp_particle_type *) spObject_device_((spObject*) pg),
//				NUM_OF_PIC);
//	}

	cudaStream_t s_local;
	cudaStreamCreate(&s_local);
	spInitializeParticle_BorisYee_Kernel<<<ctx->private_block, ctx->threadsPerBlock, 0, s_local>>>(sp->buckets);

//	for (int i = 0, ie = ctx->number_of_shared_blocks; i < ie; ++i)
//	{
//		cudaStreamSynchronize(s_shared[i]); //wait for boundary
//
//	}
//
	spSyncParticle(ctx, sp);

	cudaDeviceSynchronize(); //wait for iteration to finish
}

/******************************************************************************************/
__constant__ Real cmr_dt;
__constant__ float3 inv_dv;
__constant__ size_type entity_size_in_byte;

__constant__ int3 mesh_offset;
__constant__ int NUM_OF_ENTITY_IN_GRID;

__constant__ int3 I_OFFSET[27];
__constant__ unsigned int I_OFFSET_flag[27];

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

	*v = (f[ IX + IY + IZ /*    */] * (rx - ll) * (ry - ll) * (rz - ll)
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

		int g_f_num = ((blockDim.x + threadIdx.x + gridDim.x - RADIUS) % gridDim.x)
				+ (((blockDim.y + threadIdx.y + gridDim.y - RADIUS) % gridDim.y)
						+ ((blockDim.z + threadIdx.z + gridDim.z - RADIUS) % gridDim.z) * gridDim.y) * gridDim.x;

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

//		struct boris_page_s * pd = (struct boris_page_s *) src->data;

		for (int s = threadIdx.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += blockDim.x)
		{
#define ax _ax[threadIdx.x]
#define ay _ay[threadIdx.x]
#define az _az[threadIdx.x]

#define tx _tx[threadIdx.x]
#define ty _ty[threadIdx.x]
#define tz _tz[threadIdx.x]

#define vx _vx[s]
#define vy _vy[s]
#define vz _vz[s]

#define tt _tt[threadIdx.x]

			{
				Real rx = pg->r[0][s], ry = pg->r[1][s], rz = pg->r[2][s];

				cache_gather(&ax, tE + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
				cache_gather(&ay, tE + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
				cache_gather(&az, tE + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

				cache_gather(&tx, tB + 8 * 0, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
				cache_gather(&ty, tB + 8 * 1, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
				cache_gather(&tz, tB + 8 * 2, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);
			}

			vx = pg->v[0][s];
			vy = pg->v[1][s];
			vz = pg->v[2][s];

			pg->r[0][s] += vx * 0.5* inv_dv.x;
			pg->r[1][s] += vy * 0.5* inv_dv.y;
			pg->r[2][s] += vz * 0.5* inv_dv.z;

			vx += ax;
			vy += ay;
			vz += az;

			Real v_x, v_y, v_z;
			v_x = vx + (vy * tz- vz * ty);
			v_y = vy + (vz * tx- vx * tz);
			v_z = vz + (vx * ty- vy * tx);

			tt = 2.0 / (tx* tx+ ty * ty + tz * tz + 1.0);

			vx += ax+(v_y * tz- v_z * ty) * tt;
			vy += ax+(v_z * tx- v_x * tz) * tt;
			vz += ax+(v_x * ty- v_y * tx) * tt;

			pg->r[0][s] += vx * 0.5 * inv_dv.x;
			pg->r[0][s] += vy * 0.5 * inv_dv.y;
			pg->r[0][s] += vz * 0.5 * inv_dv.z;

			pg->v[0][s] = vx;
			pg->v[1][s] = vy;
			pg->v[2][s] = vz;

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

__global__ void spUpdateParticle_sort_Boris_kernel(spPage ** buckets_in)
{

	assert(blockDim.x * blockDim.y * blockDim.z<=NUMBER_OF_THREADS_PER_BLOCK);

#define MESH_ID (blockDim.x+mesh_offset.x + (blockDim.y+mesh_offset.y +( blockDim.z+mesh_offset.z) * gridDim.y) * gridDim.x)

	__shared__ bucket_entity_flag_t src_flag;
//	__shared__ bucket_entity_flag_t dest_flag;
	__shared__ unsigned int dest_tail;

	dest_tail = 0;

	__syncthreads();
	struct boris_page_s * dest = (struct boris_page_s *) buckets_in[MESH_ID];
	for (int i = 26; i >= 0; --i)
	{

		struct boris_page_s * pg = (struct boris_page_s *) buckets_in[((blockDim.x + I_OFFSET[i].x + gridDim.x)
				% gridDim.x)
				+ (((blockDim.y + I_OFFSET[i].y + gridDim.y) % gridDim.y)
						+ ((blockDim.z + I_OFFSET[i].z + gridDim.z) % gridDim.z) * gridDim.y) * gridDim.x];

		while (pg != 0x0)
		{
			if (threadIdx.x == 0)
			{
				src_flag = pg->flag;

			}

			for (int s = threadIdx.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += blockDim.x)
			{

				if ((src_flag) & 0x3F == I_OFFSET_flag[i])
				{

					unsigned int tail = atomicAdd(&dest_tail, 1);

					assert(tail<SP_NUMBER_OF_ENTITIES_IN_PAGE);

					dest->r[0][tail] = pg->r[0][s];
					dest->r[1][tail] = pg->r[1][s];
					dest->r[2][tail] = pg->r[2][s];

					dest->v[0][tail] = pg->v[0][s];
					dest->v[1][tail] = pg->v[1][s];
					dest->v[2][tail] = pg->v[2][s];

					dest->f[tail] = pg->f[s];
					dest->w[tail] = pg->w[s];

				}

			}
			__syncthreads();

			pg = (struct boris_page_s *) (pg->next);
		}

	}

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

		bucket_entity_flag_t dest_flag = pg->flag;

		for (int s = threadIdx.x; s < SP_NUMBER_OF_ENTITIES_IN_PAGE; s += blockDim.x)
		{
			if (dest_flag & (0x3F << (s * 6)) == 0x15)
			{
				Real w0 = abs((pg->r[0][s] - 0.5) * (pg->r[1][s] - 0.5) * (pg->r[2][s] - 0.5)) * pg->f[s] * pg->w[s];

				J4.w += w0;
				J4.x += w0 * pg->v[0][s];
				J4.y += w0 * pg->v[1][s];
				J4.z += w0 * pg->v[2][s];
			}
			else
			{
				dest_flag = dest_flag & ((0x3F << (s * 6)));
			}
		}

		atomicAnd(&(((struct boris_page_s *) buckets[MESH_ID])->flag), ~(dest_flag));

		pg = (struct boris_page_s *) (pg->next);
	}

	atomicAdd(&(fJ[MESH_ID + NUM_OF_ENTITY_IN_GRID * 0]), J4.x);
	atomicAdd(&(fJ[MESH_ID + NUM_OF_ENTITY_IN_GRID * 1]), J4.y);
	atomicAdd(&(fJ[MESH_ID + NUM_OF_ENTITY_IN_GRID * 2]), J4.z);
	atomicAdd(&(fRho[MESH_ID]), J4.w);

}

void spUpdateParticle_BorisYee(spMesh *ctx, Real dt, sp_particle_type *pg, const sp_field_type *fE,
		const sp_field_type *fB, sp_field_type *fRho, sp_field_type *fJ)
{

	float3 t_inv_dv = make_float3(dt / ctx->dx.x, dt / ctx->dx.y, dt / ctx->dx.z);
	Real t_cmr_dt = 0.5 * dt * pg->charge / pg->mass;
	size_type t_entity_size_in_byte = pg->entity_size_in_byte;

	cudaMemcpyToSymbol(&inv_dv, &t_inv_dv, sizeof(float3), cudaMemcpyDefault);
	cudaMemcpyToSymbol(&cmr_dt, &t_cmr_dt, sizeof(Real), cudaMemcpyDefault);
	cudaMemcpyToSymbol(&entity_size_in_byte, &t_entity_size_in_byte, sizeof(size_type), cudaMemcpyDefault);

	spUpdateParticle_push_Boris_Kernel<<<ctx->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(pg->buckets,
			((Real*) fE->device_data), ((Real*) fB->device_data));

	spUpdateParticle_sort_Boris_kernel<<<ctx->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(pg->buckets);

	spUpdateParticle_scatter_Boris_kernel<<<ctx->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(pg->buckets,
			((Real*) fRho->device_data), ((Real*) fJ->device_data));

	cudaDeviceSynchronize();

	spSyncParticle(ctx, pg);
	spSyncField(ctx, fJ);
	spSyncField(ctx, fRho);

}
/***************************************************************************************************************/

__global__ void spUpdateField_Yee_kernel(const Real *fJ, Real *fE, Real *fB)
{
}
void spUpdateField_Yee(spMesh *ctx, Real dt, const sp_field_type *fRho, const sp_field_type *fJ, sp_field_type *fE,
		sp_field_type *fB)
{
	spUpdateField_Yee_kernel<<<ctx->dims, NUMBER_OF_THREADS_PER_BLOCK>>>(((Real*) fJ->device_data),
			((Real*) fE->device_data), ((Real*) fB->device_data));

	cudaDeviceSynchronize();        //wait for iteration to finish
	spSyncField(ctx, fE);
	spSyncField(ctx, fB);
}

