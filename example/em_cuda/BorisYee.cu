//
// Created by salmon on 16-6-14.
//

#include "Boris.h"
#include <stdio.h>
#include "../../sp_lite/sp_def.h"
#include "../../sp_lite/spField.h"
#include "../../sp_lite/spMesh.h"
#include "../../sp_lite/spParticle.h"
#include "../../sp_lite/spPage.h"

#include "../../sp_lite/spMesh.cu"
#include "../../sp_lite/spField.cu"
#include "../../sp_lite/spParticle.cu"


#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y

/******************************************************************************************/
#define ll 0
#define rr 1.0
MC_HOST_DEVICE
void cache_gather(Real *v, Real const f[], Real const *r0, const Real *r1)
{
	Real r[3] =
	{ r0[0] - r1[0], r0[1] - r1[1], r0[2] - r1[2] };
	id_type s = (int) (r[0]) * IX + (int) (r[1]) * IY + (int) (r[2]) * IZ;

	*v = f[s + IX + IY + IZ /*  */] * (r[0] - ll) * (r[1] - ll) * (r[2] - ll)
			+ f[s + IX + IY /*     */] * (r[0] - ll) * (r[1] - ll) * (rr - r[2])
			+ f[s + IX + IZ /*     */] * (r[0] - ll) * (rr - r[1]) * (r[2] - ll)
			+ f[s + IX /*          */] * (r[0] - ll) * (rr - r[1]) * (rr - r[2])
			+ f[s + IY + IZ /*     */] * (rr - r[0]) * (r[1] - ll) * (r[2] - ll)
			+ f[s + IY /*          */] * (rr - r[0]) * (r[1] - ll) * (rr - r[2])
			+ f[s + IZ /*          */] * (rr - r[0]) * (rr - r[1]) * (r[2] - ll)
			+ f[s /*               */] * (rr - r[0]) * (rr - r[1]) * (rr - r[2]);
}
MC_HOST_DEVICE
void cache_scatter(Real f[], Real v, Real const *r0, Real const *r1)
{
	Real r[3] =
	{ r0[0] - r1[0], r0[1] - r1[1], r0[2] - r1[2] };
	id_type s = (int) (r[0]) * IX + (int) (r[1]) * IY + (int) (r[2]) * IZ;

	f[s + IX + IY + IZ /*  */] += v * (r[0] - ll) * (r[1] - ll) * (r[2] - ll);
	f[s + IX + IY /*       */] += v * (r[0] - ll) * (r[1] - ll) * (rr - r[2]);
	f[s + IX + IZ /*       */] += v * (r[0] - ll) * (rr - r[1]) * (r[2] - ll);
	f[s + IX /*            */] += v * (r[0] - ll) * (rr - r[1]) * (rr - r[2]);
	f[s + IY + IZ /*       */] += v * (rr - r[0]) * (r[1] - ll) * (r[2] - ll);
	f[s + IY /*            */] += v * (rr - r[0]) * (r[1] - ll) * (rr - r[2]);
	f[s + IZ /*            */] += v * (rr - r[0]) * (rr - r[1]) * (r[2] - ll);
	f[s/*                  */] += v * (rr - r[0]) * (rr - r[1]) * (rr - r[2]);

}

#undef ll
#undef rr
#undef IX
#undef IY
#undef IZ
#define _R 1.0
MC_CONSTANT Real id_to_shift_[][3] =
{ //
		{ 0, 0, 0 },           // 000
				{ _R, 0, 0 },           // 001
				{ 0, _R, 0 },           // 010
				{ 0, 0, _R },          // 011
				{ _R, _R, 0 },           // 100
				{ _R, 0, _R },          // 101
				{ 0, _R, _R },          // 110
				{ 0, _R, _R },          // 111
		};
MC_CONSTANT int sub_index_to_id_[4][3] =
{ //
		{ 0, 0, 0 }, /*VERTEX*/
		{ 1, 2, 4 }, /*EDGE*/
		{ 6, 5, 3 }, /*FACE*/
		{ 7, 7, 7 } /*VOLUME*/

		};
MC_CONSTANT int cache_cell_offset_tag[CACHE_SIZE] =
{ };
MC_CONSTANT size_type cache_cell_offset[CACHE_SIZE] =
{ };
#undef _R

MC_HOST_DEVICE
inline void spBorisPushOne(struct boris_point_s const *p0, struct boris_point_s *p1, Real dt, Real q, Real m,
		Real const tE[3][CACHE_SIZE], Real const tB[3][CACHE_SIZE], Real tJ[4][CACHE_SIZE], const Real *inv_dx)
{

	Real E[3], B[3];

	cache_gather(&E[0], tE[0], p0->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
	cache_gather(&E[1], tE[1], p0->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
	cache_gather(&E[2], tE[2], p0->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

	cache_gather(&B[0], tB[0], p0->r, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
	cache_gather(&B[1], tB[1], p0->r, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
	cache_gather(&B[2], tB[2], p0->r, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

	p1->r[0] = p0->r[0] + p0->v[0] * dt * 0.5 * inv_dx[0];
	p1->r[1] = p0->r[1] + p0->v[1] * dt * 0.5 * inv_dx[1];
	p1->r[2] = p0->r[2] + p0->v[2] * dt * 0.5 * inv_dx[2];

	Real v_[3], t[3];

	t[0] = B[0] * (q / m * dt * 0.5);
	t[1] = B[1] * (q / m * dt * 0.5);
	t[2] = B[2] * (q / m * dt * 0.5);

	p1->v[0] = p0->v[0] + E[0] * (q / m * dt * 0.5);
	p1->v[1] = p0->v[1] + E[1] * (q / m * dt * 0.5);
	p1->v[2] = p0->v[2] + E[2] * (q / m * dt * 0.5);

	v_[0] = p1->v[0] + (p1->v[1] * t[2] - p1->v[2] * t[1]);
	v_[1] = p1->v[1] + (p1->v[2] * t[0] - p1->v[0] * t[2]);
	v_[2] = p1->v[2] + (p1->v[0] * t[1] - p1->v[1] * t[0]);

	Real tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + 1.0;

	p1->v[0] += (v_[1] * t[2] - v_[2] * t[1]) * 2.0 / tt;
	p1->v[1] += (v_[2] * t[0] - v_[0] * t[2]) * 2.0 / tt;
	p1->v[2] += (v_[0] * t[1] - v_[1] * t[0]) * 2.0 / tt;

	p1->v[0] += E[0] * (q / m * dt * 0.5);
	p1->v[1] += E[1] * (q / m * dt * 0.5);
	p1->v[2] += E[2] * (q / m * dt * 0.5);

	p1->r[0] += p1->v[0] * dt * 0.5 * inv_dx[0];
	p1->r[1] += p1->v[1] * dt * 0.5 * inv_dx[1];
	p1->r[2] += p1->v[2] * dt * 0.5 * inv_dx[2];

	cache_scatter(tJ[0], p1->f * p1->w * q, p1->r, id_to_shift_[sub_index_to_id_[0/*VERTEX*/][0]]);
	cache_scatter(tJ[1], p1->f * p1->w * p1->v[0] * q, p1->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
	cache_scatter(tJ[2], p1->f * p1->w * p1->v[1] * q, p1->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
	cache_scatter(tJ[3], p1->f * p1->w * p1->v[2] * q, p1->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

}
/******************************************************************************************/

__global__ void spInitializeParticle_BorisYee_Kernel(spMesh *ctx, sp_particle_type *p, size_type NUM_OF_PIC)
{

	int pos = blockIdx.x + (blockIdx.y * blockDim.x + blockIdx.z) * blockDim.y;

	if (threadIdx.x == 0)
	{
	}
	__syncthreads();
	{
		((spEntity*) (p->buckets[pos]->data + threadIdx.x * p->entity_size_in_byte))->tag = threadIdx.x;
	}

}
/******************************************************************************************/

__global__ void spUpdateParticle_BorisYee_Kernel(spMesh *m, Real dt, sp_particle_type *sp, const Real *fE,
		const Real *fB, Real *fRho, Real *fJ)
{
	size_type entity_size_in_byte = sp->entity_size_in_byte;
	Real mass = sp->mass;
	Real charge = sp->charge;

	MC_SHARED Real tE[3][CACHE_SIZE], tB[3][CACHE_SIZE], tJ[4][CACHE_SIZE];

	int pos = blockIdx.x + (blockIdx.y * blockDim.x + blockIdx.z) * blockDim.y;
	id_type cell_idx = m->cell_idx[pos];
	size_type sub_idx = threadIdx.x;

	spPage * write_cache = spParticleCreateBucket(sp, 2);

// read tE,tB from E,B
// clear tJ

// TODO load data to cache
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < CACHE_SIZE; ++j)
		{
			tJ[i][j] = 0.0;
		}

	for (int n = 0; n < CACHE_SIZE; ++n)
	{
		id_type tag = cache_cell_offset_tag[n];
		spPage *pg = sp->buckets[cell_idx + cache_cell_offset[n]];
		while (pg != 0x0)
		{
			boris_point_s *p0 = (boris_point_s *) (pg->data + sub_idx * entity_size_in_byte);

			if ((pg->tag & (0x1 << sub_idx) != 0) && (p0->tag & 0x3F) == tag)
			{

				spBorisPushOne(p0, (boris_point_s *) spEntityInsert(write_cache, entity_size_in_byte),		//
				dt, charge, mass, tE, tB, tJ, m->inv_dx);

			}

			pg = pg->next;
		}        //	while (pg != 0x0)
	}        //	for (int n = 0; n < CACHE_SIZE; ++n)
	__syncthreads();
//TODO copy write_cache to memory

//TODO atomic_add tJ to fJ
#pragma unroll
	for (int s = 0; s < CACHE_SIZE; ++s)
	{
//				size_type idx = posFromCacheIdx(s, ctx->i_dims);
//				atomicAdd(&(fRho[idx]), tJ[0][idx]);
//				atomicAdd(&(fJ[idx * 3 + 0]), tJ[1][idx]);
//				atomicAdd(&(fJ[idx * 3 + 0]), tJ[2][idx]);
//				atomicAdd(&(fJ[idx * 3 + 0]), tJ[3][idx]);
	}
	__syncthreads();

}

void spInitializeParticle_BorisYee(spMesh *ctx, sp_particle_type *pg, size_type NUM_OF_PIC)
{
//	spInitializeParticle_BorisYee_Kernel<<<ctx->numBlocks, ctx->threadsPerBlock>>>(ctx, pg, NUM_OF_PIC);

	cudaStream_t s_shared[ctx->number_of_shared_blocks];

	for (int i = 0, ie = ctx->number_of_shared_blocks; i < ie; ++i)
	{
		cudaStreamCreate(&s_shared[i]);
		spInitializeParticle_BorisYee_Kernel<<<ctx->shared_blocks[i], ctx->threadsPerBlock, 0, s_shared[i]>>>(
				(spMesh *) spObject_device_((spObject*) ctx), (sp_particle_type *) spObject_device_((spObject*) pg),
				NUM_OF_PIC);
	}
	cudaStream_t s_local;
	cudaStreamCreate(&s_local);
	spInitializeParticle_BorisYee_Kernel<<<ctx->private_block, ctx->threadsPerBlock, 0, s_local>>>(
			(spMesh *) spObject_device_((spObject*) ctx), (sp_particle_type *) spObject_device_((spObject*) pg),
			NUM_OF_PIC);
	for (int i = 0, ie = ctx->number_of_shared_blocks; i < ie; ++i)
	{
		cudaStreamSynchronize(s_shared[i]); //wait for boundary

	}

	spSyncParticle(ctx, pg);

	cudaDeviceSynchronize(); //wait for iteration to finish
}

void spUpdateParticle_BorisYee(spMesh *ctx, Real dt, sp_particle_type *pg, const sp_field_type *fE,
		const sp_field_type *fB, sp_field_type *fRho, sp_field_type *fJ)
{
//	cudaStream_t s1;
//	cudaStreamCreate(&s1);
//	cudaStream_t s2;
//	cudaStreamCreate(&s2);
//
//	spUpdateParticle_BorisYee_Kernel<<<ctx->numBlocks, ctx->threadsPerBlock, 0, s1>>>(
//			(spMesh *) spObject_device_((spObject*) ctx),
//			dt, //
//			(sp_particle_type *) spObject_device_((spObject*) pg),
//			(const sp_field_type *) spObject_device_((spObject*) fE),
//			(const sp_field_type *) spObject_device_((spObject*) fB),
//			(sp_field_type *) spObject_device_((spObject*) fRho), (sp_field_type *) spObject_device_((spObject*) fJ));
//
//	spUpdateParticle_BorisYee_Kernel<<<ctx->numBlocks, ctx->threadsPerBlock, 0, s2>>>(
//			(spMesh *) spObject_device_((spObject*) ctx),
//			dt, //
//			(sp_particle_type *) spObject_device_((spObject*) pg),
//			(const sp_field_type *) spObject_device_((spObject*) fE),
//			(const sp_field_type *) spObject_device_((spObject*) fB),
//			(sp_field_type *) spObject_device_((spObject*) fRho), (sp_field_type *) spObject_device_((spObject*) fJ));
//	cudaStreamSynchronize(s1); //wait for boundary
//
//	spSyncParticle(ctx, pg);
//	spSyncField(ctx, fJ);
//	spSyncField(ctx, fRho);
//	cudaDeviceSynchronize(); //wait for iteration to finish

}
/***************************************************************************************************************/
//__global__ void spUpdateField_Yee_kernel(spMesh *ctx, Real dt, const sp_field_type *fRho, const sp_field_type *fJ,
//		sp_field_type *fE, sp_field_type *fB)
__global__ void spUpdateField_Yee_kernel(spMesh *ctx, Real dt, const Real *fRho, const Real *fJ, Real *fE, Real *fB)
{
	index_type ix = (blockIdx.x * blockDim.x + threadIdx.x);
	index_type iy = (blockIdx.y * blockDim.y + threadIdx.y);
	index_type iz = (blockIdx.z * blockDim.z + threadIdx.z);
	size_type dim_x = gridDim.x * blockDim.x;
	size_type dim_y = gridDim.y * blockDim.y;
	size_type dim_z = gridDim.z * blockDim.z;

	int n = ix + (iy + iz * dim_y) * dim_z;

	(fE)[n * 3 + 0] = ix;
	(fE)[n * 3 + 1] = iy;
	(fE)[n * 3 + 2] = iz;

}
void spUpdateField_Yee(spMesh *ctx, Real dt, const sp_field_type *fRho, const sp_field_type *fJ, sp_field_type *fE,
		sp_field_type *fB)
{

//	cudaStream_t s_shared[ctx->number_of_shared_blocks];
//
//	for (int i = 0, ie = ctx->number_of_shared_blocks; i < ie; ++i)
//	{
//		cudaStreamCreate(&s_shared[i]);
//
//		spUpdateField_Yee_kernel<<<ctx->shared_blocks[i], ctx->threadsPerBlock, 0, s_shared[i]>>>(
//				(spMesh *) spObject_device_((spObject*) ctx),
//				dt, //
//				(const sp_field_type *) spObject_device_((spObject*) fRho),
//				(const sp_field_type *) spObject_device_((spObject*) fJ),
//				(sp_field_type *) spObject_device_((spObject*) fE), (sp_field_type *) spObject_device_((spObject*) fB));
//	}
//	cudaStream_t s_local;
//	cudaStreamCreate(&s_local);
//
//	spUpdateField_Yee_kernel<<<ctx->private_block, 1>>>((spMesh *) spObject_device_((spObject*) ctx),
//			dt, //
//			(const sp_field_type *) spObject_device_((spObject*) fRho),
//			(const sp_field_type *) spObject_device_((spObject*) fJ),
//			(sp_field_type *) spObject_device_((spObject*) fE), (sp_field_type *) spObject_device_((spObject*) fB));

	dim3 grid_dim = ctx->private_block;
	grid_dim.x = ctx->private_block.x / ctx->threadsPerBlock.x;
	grid_dim.y = ctx->private_block.y / ctx->threadsPerBlock.y;
	grid_dim.z = ctx->private_block.z / ctx->threadsPerBlock.z;
	spUpdateField_Yee_kernel<<<grid_dim, ctx->threadsPerBlock>>>((spMesh *) spObject_device_((spObject*) ctx), dt, //
			((Real*) fRho->device_data), ((Real*) fJ->device_data), ((Real*) fE->device_data), ((Real*) fB->device_data));
//	for (int i = 0, ie = ctx->number_of_shared_blocks; i < ie; ++i)
//	{
//		cudaStreamSynchronize(s_shared[i]); //wait for boundary
//	}
//
//	spSyncField(ctx, fE);
//	spSyncField(ctx, fB);

	cudaDeviceSynchronize(); //wait for iteration to finish

}

