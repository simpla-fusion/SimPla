//
// Created by salmon on 16-6-14.
//

#include "../../src/capi/sp_def.h"
#include "Boris.h"
#include "BorisYee.h"
#include <stdio.h>
#include "../../src/capi/spField.h"
#include "../../src/capi/spMesh.h"
#include "../../src/capi/spParticle.h"
#include "../../src/capi/spBucketFunction.h"

MC_HOST_DEVICE_PREFIX void
spBorisPushOne (struct boris_point_s const *p, struct boris_point_s *p_next,
				Real cmr, Real dt, Real const E[3], Real const B[3],
				const Real *inv_dx);

MC_HOST_DEVICE_PREFIX Real
spBorisGetRho (struct boris_point_s const *p);

MC_HOST_DEVICE_PREFIX Real
spBorisGetJ (struct boris_point_s const *p, int n);

MC_HOST_DEVICE_PREFIX Real
spBorisGetE (struct boris_point_s const *p);

#define CACHE_EXTENT_X 4
#define CACHE_EXTENT_Y 4
#define CACHE_EXTENT_Z 4
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)

#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y

__global__ void
spInitializeParticle_BorisYee_Kernel (spMesh *ctx, sp_particle_type *pg)
{

}

#define ll 0
#define rr 1.0

MC_DEVICE void
cache_gather (Real *v, Real const f[CACHE_SIZE], Real const *r0, const Real *r1)
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

MC_DEVICE void
cache_scatter (Real f[CACHE_SIZE], Real v, Real const *r0, Real const *r1)
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

/**
 *\verbatim
 *                ^y
 *               /
 *        z     /
 *        ^    /
 *    PIXEL0 110-------------111 VOXEL
 *        |  /|              /|
 *        | / |             / |
 *        |/  |    PIXEL1  /  |
 * EDGE2 100--|----------101  |
 *        | m |           |   |
 *        |  010----------|--011 PIXEL2
 *        |  / EDGE1      |  /
 *        | /             | /
 *        |/              |/
 *       000-------------001---> x
 *                       EDGE0
 *
 *\endverbatim
 */

/* @formatter:off*/
#define _R 1.0
MC_CONSTANT Real id_to_shift_[][3] =
  { //
	  { 0, 0, 0 },           // 000
		  { _R, 0, 0 },           // 001
		  { 0,
		  _R, 0 },           // 010
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
  {

  };
MC_CONSTANT size_type cache_cell_offset[CACHE_SIZE] =
  {

  };
#undef _R

/* @formatter:on*/
__global__ void
spUpdateParticle_BorisYee_Kernel (spMesh *m, sp_particle_type *sp, Real dt,
								  const sp_field_type * fE,
								  const sp_field_type * fB,
								  sp_field_type * fRho, sp_field_type * fJ)
{
  size_type entity_size_in_byte = sp->entity_size_in_byte;
  Real cmr = sp->charge / sp->mass;
  Real charge = sp->charge;

  Real tE[3][CACHE_SIZE], tB[3][CACHE_SIZE], tJ[4][CACHE_SIZE];

  MC_SHARED int write_cache_count;
  MC_SHARED byte_type write_cache[sizeof(boris_point_s)
	  * SP_NUMBER_OF_ENTITIES_IN_PAGE * 2];

  for (size_type _blk_s = blockIdx.x, _blk_e = m->number_of_idx;
	  _blk_s < _blk_e; _blk_s += blockDim.x)
	{
	  size_type cell_idx = m->cell_idx[_blk_s];

	  // read tE,tB from E,B
	  // clear tJ

	  write_cache_count = 0;

	  __syncthreads ();

	  // TODO load data to cache

	  for (int n = 0; n < CACHE_SIZE; ++n)
		{
		  id_type tag = cache_cell_offset_tag[n];
		  bucket_type pg = sp->buckets[cell_idx + cache_cell_offset[n]];
		  while (pg != 0x0)
			{
			  boris_point_s *p0 = (boris_point_s *) (pg->data
				  + threadIdx.x * entity_size_in_byte);

			  if ((pg->tag & (0x1 << threadIdx.x) != 0)
				  && (p0->_tag & 0x3F) == tag)
				{
				  boris_point_s * p1 = (boris_point_s *) (write_cache
					  + atomicAdd (&write_cache_count, 1) * entity_size_in_byte);

				  Real E[3], B[3];

				  cache_gather (&E[0], tE[0], p0->r,
								id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
				  cache_gather (&E[1], tE[1], p0->r,
								id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
				  cache_gather (&E[2], tE[2], p0->r,
								id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

				  cache_gather (&B[0], tB[0], p0->r,
								id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
				  cache_gather (&B[1], tB[1], p0->r,
								id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
				  cache_gather (&B[2], tB[2], p0->r,
								id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

				  spBorisPushOne (p0, p1, cmr, dt, E, B, m->inv_dx);

				  cache_scatter (
					  tJ[0], spBorisGetRho (p1) * charge, p1->r,
					  id_to_shift_[sub_index_to_id_[0/*VERTEX*/][0]]);
				  cache_scatter (tJ[1], spBorisGetJ (p1, 0) * charge, p1->r,
								 id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
				  cache_scatter (tJ[2], spBorisGetJ (p1, 1) * charge, p1->r,
								 id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
				  cache_scatter (tJ[3], spBorisGetJ (p1, 2) * charge, p1->r,
								 id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

				  /****************************************************************/

				}

			  pg = pg->next;
			}		//	while (pg != 0x0)
		}		//	for (int n = 0; n < CACHE_SIZE; ++n)

	  __syncthreads ();

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

	}
}

__global__ void
spUpdateField_Yee_kernel (spMesh *ctx, Real dt, const sp_field_type* fRho,
						  const sp_field_type *fJ, sp_field_type* fE,
						  sp_field_type *fB)
{

}

void
spInitializeParticle_BorisYee (spMesh *ctx, sp_particle_type *pg)
{
  spInitializeParticle_BorisYee_Kernel <<<ctx->numBlocks, ctx->threadsPerBlock>>> (
	  ctx, pg);
}

void
spUpdateParticle_BorisYee (spMesh *ctx, sp_particle_type *pg, Real dt,
						   const sp_field_type * fE, const sp_field_type * fB,
						   sp_field_type * fRho, sp_field_type * fJ)
{
  //number_of_core / SP_NUMBER_OF_ELEMENT_IN_PAGE

  spUpdateParticle_BorisYee_Kernel <<<ctx->numBlocks, ctx->threadsPerBlock>>> (
	  ctx, pg, dt, fE, fB, fRho, fJ);

}

void
spUpdateField_Yee (spMesh *ctx, Real dt, const sp_field_type* fRho,
				   const sp_field_type* fJ, sp_field_type* fE,
				   sp_field_type* fB)
{
  spUpdateField_Yee_kernel <<<ctx->numBlocks, ctx->threadsPerBlock>>> (ctx, dt,
																	   fRho, fJ,
																	   fE, fB);
}

