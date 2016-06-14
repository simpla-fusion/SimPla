//
// Created by salmon on 16-6-14.
//

#include "BorisYee.h"
#include "Boris.h"

#include "../../src/sp_config.h"
#include "../../src/particle/BucketContainer.h"
//#include "../../src/particle/ParticleUtility.h

#include "../../src/particle/BucketContainer.c"

#define CACHE_EXTENT_X 4
#define CACHE_EXTENT_Y 4
#define CACHE_EXTENT_Z 4
#define CACHE_SIZE (CACHE_EXTENT_X*CACHE_EXTENT_Y*CACHE_EXTENT_Z)

#define IX  1
#define IY  CACHE_EXTENT_X
#define IZ  CACHE_EXTENT_X*CACHE_EXTENT_Y

#define ll 0
#define rr 1.0

__device__  void cache_gather(Real *v, Real const f[CACHE_SIZE], Real const *r0,
		const Real *r1)
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
			+ f[s /*               */] * (rr - r[0]) * (rr - r[1])
					* (rr - r[2]);
}

__device__ void cache_scatter(Real f[CACHE_SIZE], Real v, Real const *r0,
		Real const *r1)
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
__device__ Real id_to_shift_[][3] =
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
__device__ int sub_index_to_id_[4][3] =
{ //
		{ 0, 0, 0 }, /*VERTEX*/
		{ 1, 2, 4 }, /*EDGE*/
		{ 6, 5, 3 }, /*FACE*/
		{ 7, 7, 7 } /*VOLUME*/

		};

#undef _R

/* @formatter:on*/
__device__
void spBorisYeeUpdate_kernel(BorisYeeUpdateArgs const *args, Real dt,
		spPage **first, spPage **second, spPagePool *pool, const Real *fE,
		const Real *fB, Real *fRho, Real *fJ)
{
	size_type entity_size_in_byte = (*first)->entity_size_in_byte;

	__shared__ Real tE[3][CACHE_SIZE], tB[3][CACHE_SIZE], tJ[4][CACHE_SIZE];
	__shared__ spPage *read_buffer[CACHE_SIZE];
//	__shared__ spPage *write_buffer[CACHE_SIZE];
//	__shared__ bucket_page_status_flag_t shift_flag[CACHE_SIZE];

	for (size_type _blk_s = blockIdx.x, _blk_e = args->number_of_idx;
			_blk_s < _blk_e; _blk_s += blockDim.x)
	{
		size_type cell_idx = args->cell_idx[_blk_s];

		// read tE,tB from E,B
		// clear tJ
		spPage **src = &first[cell_idx];
		spPage **dest = &second[cell_idx];
		id_type tag = 0x0;
		size_type num = spEntityCountIf(*src, tag);

		__shared__ byte_type write_buffer[sizeof(boris_point_s)
				* SP_NUMBER_OF_ENTITIES_IN_PAGE];
//		write_buffer=malloc( num * SP_NUMBER_OF_ENTITIES_IN_PAGE);

		// TODO load data to cache

		for (int n = 0; n < CACHE_SIZE; ++n)
		{

			spPage *pg = read_buffer[n];

			bucket_page_status_flag_t dest_flag = 0x0;

			bucket_page_status_flag_t read_flag = 0x0;

			bucket_page_status_flag_t write_flag = 0x0;

			for (boris_point_s *p0, *p1 = 0x0; (p0 =
					(boris_point_s *) spEntityNext(&pg, &read_flag)) != 0x0;)
			{
				if ((p0->_tag & 0x3F) == tag)
				{
					if (write_flag == 0x0 || write_buffer == 0x0)
					{
						/** atomic */

						/** atomic */
						spPage *t = spPageCreate(1, pool);

//						cudaMemcpy(t->data, write_buffer,
//						SP_NUMBER_OF_ENTITIES_IN_PAGE * entity_size_in_byte,
//								cudaMemcpyDeviceToDevice);

						t->flag = dest_flag;
						spPagePushFront(dest, t);
						dest_flag = 0x0;
						write_flag = 0x0;
						p1 = (boris_point_s *) write_buffer;
					}
					/****************************************************************/

					Real E[3], B[3];

					cache_gather(&E[0], tE[0], p0->r,
							id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
					cache_gather(&E[1], tE[1], p0->r,
							id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
					cache_gather(&E[2], tE[2], p0->r,
							id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

					cache_gather(&B[0], tB[0], p0->r,
							id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
					cache_gather(&B[1], tB[1], p0->r,
							id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
					cache_gather(&B[2], tB[2], p0->r,
							id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

					spBorisPushOne(p0, p1, args->cmr, dt, E, B, args->inv_dx);

					cache_scatter(tJ[0], spBorisGetRho(p1), p1->r,
							id_to_shift_[sub_index_to_id_[0/*VERTEX*/][0]]);
					cache_scatter(tJ[1], spBorisGetJ(p1, 0), p1->r,
							id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
					cache_scatter(tJ[2], spBorisGetJ(p1, 1), p1->r,
							id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
					cache_scatter(tJ[3], spBorisGetJ(p1, 2), p1->r,
							id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

					/****************************************************************/
					p1->_tag &= ~(0x3F); // clear tag
					dest_flag |= write_flag;
					write_flag <<= 1;
					p1 += entity_size_in_byte;

				}
			}
			__syncthreads();

			//TODO atomic_add tJ to fJ
			for (int s = 0; s < CACHE_SIZE; ++s)
			{
//				size_type idx = posFromCacheIdx(s, args->i_dims);
//				atomicAdd(&(fRho[idx]), tJ[0][idx]);
//				atomicAdd(&(fJ[idx * 3 + 0]), tJ[1][idx]);
//				atomicAdd(&(fJ[idx * 3 + 0]), tJ[2][idx]);
//				atomicAdd(&(fJ[idx * 3 + 0]), tJ[3][idx]);
			}

		}        //foreach block

	}
}

void spBorisYeeUpdate(BorisYeeUpdateArgs const *args, Real dt, spPage **first,
		spPage **second, spPagePool *pool, const Real *fE, const Real *fB,
		Real *fRho, Real *fJ)
{
	//number_of_core / SP_NUMBER_OF_ELEMENT_IN_PAGE
//	int numBlocks = (16);
//
//	dim3 threadsPerBlock(SP_NUMBER_OF_ENTITIES_IN_PAGE, 1);
//
//	spBorisYeeUpdate_kernel<<<numBlocks, threadsPerBlock>>>(args, dt, first,
//			second, pool, fE, fB, fRho, fJ);

}
