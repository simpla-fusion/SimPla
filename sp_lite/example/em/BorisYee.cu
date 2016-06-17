//
// Created by salmon on 16-6-14.
//

#include "Boris.h"
#include <stdio.h>
#include "../../src/sp_def.h"
#include "../../src/spField.h"
#include "../../src/spMesh.h"
#include "../../src/spParticle.h"
#include "../../src/spPage.h"

#include "../../src/spMesh.cu"
#include "../../src/spField.cu"
#include "../../src/spParticle.cu"

#define CACHE_EXTENT_X 4
#define CACHE_EXTENT_Y 4
#define CACHE_EXTENT_Z 4
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)

#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y

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

MC_HOST_DEVICE void
spBorisPushOne(struct boris_point_s const *p0, struct boris_point_s *p1, Real dt, Real q, Real m,
		Real const tE[3][CACHE_SIZE], Real const tB[3][CACHE_SIZE], Real tJ[4][CACHE_SIZE], const Real *inv_dx);

MC_HOST_DEVICE Real
spBorisGetRho(struct boris_point_s const *p);

MC_HOST_DEVICE Real
spBorisGetJ(struct boris_point_s const *p, int n);

MC_HOST_DEVICE Real
spBorisGetE(struct boris_point_s const *p);
MC_HOST_DEVICE
void cache_scatter(Real f[], Real v, Real const *r0, Real const *r1);
MC_HOST_DEVICE
void cache_gather(Real *v, Real const f[], Real const *r0, const Real *r1);

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
/* @formatter:on*/
__global__ void spUpdateParticle_BorisYee_Kernel(spMesh *m, sp_particle_type *sp, Real dt, const sp_field_type *fE,
		const sp_field_type *fB, sp_field_type *fRho, sp_field_type *fJ)
{
	size_type entity_size_in_byte = sp->entity_size_in_byte;
	Real mass = sp->mass;
	Real charge = sp->charge;

	MC_SHARED Real tE[3][CACHE_SIZE], tB[3][CACHE_SIZE];
	Real tJ[4][CACHE_SIZE];

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

__global__ void spUpdateField_Yee_kernel(spMesh *ctx, Real dt, const sp_field_type *fRho, const sp_field_type *fJ,
		sp_field_type *fE, sp_field_type *fB)
{

}

void spInitializeParticle_BorisYee(spMesh *ctx, sp_particle_type *pg, size_type NUM_OF_PIC)
{
	spInitializeParticle_BorisYee_Kernel<<<ctx->numBlocks, ctx->threadsPerBlock>>>(ctx, pg, NUM_OF_PIC);
}

void spUpdateParticle_BorisYee(spMesh *ctx, sp_particle_type *pg, Real dt, const sp_field_type *fE,
		const sp_field_type *fB, sp_field_type *fRho, sp_field_type *fJ)
{

	spUpdateParticle_BorisYee_Kernel<<<ctx->numBlocks, ctx->threadsPerBlock>>>(ctx, (sp_particle_type *) pg->self, dt,
			(const sp_field_type *) fE->self, (const sp_field_type *) fB->self, (sp_field_type *) fRho->self,
			(sp_field_type *) fJ->self);

}

void spUpdateField_Yee(spMesh *ctx, Real dt, const sp_field_type *fRho, const sp_field_type *fJ, sp_field_type *fE,
		sp_field_type *fB)
{
	spUpdateField_Yee_kernel<<<ctx->numBlocks, ctx->threadsPerBlock>>>(ctx, dt, fRho, fJ, fE, fB);
}

