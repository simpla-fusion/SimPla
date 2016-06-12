//
// Created by salmon on 16-6-11.
//
#include "BorisYee.h"

#include "../../src/particle/SmallObjPool.h"

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#define CUDA_HOST   __host__
#define CUDA_SHARED  __shared__
#define INLINE_PREFIX __device__ __forceinline__

#else
#define CUDA_DEVICE
#define CUDA_GLOBAL
#define CUDA_HOST
#define CUDA_SHARED
#endif

#include "Boris.h"

struct spPage;

#define IX  1
#define IY  3
#define IZ  9

#define ll -0.5
#define rr 0.5

CUDA_DEVICE
void cache_gather(Real *v, Real const f[CACHE_SIZE], Real const *r0,
		const Real *r1) {
	Real r[3] = { r0[0] - r1[0], r0[1] - r1[1], r0[2] - r1[2] };
	id_type s = (int) (r[0]) * IX + (int) (r[1]) * IY + (int) (r[2]) * IZ;

	*v = f[s + IX + IY + IZ /* */] * (r[0] - ll) * (r[1] - ll) * (r[2] - ll)
			+ f[s + IX + IY /*     */] * (r[0] - ll) * (r[1] - ll) * (rr - r[2])
			+ f[s + IX + IZ /*     */] * (r[0] - ll) * (rr - r[1]) * (r[2] - ll)
			+ f[s + IX /*          */] * (r[0] - ll) * (rr - r[1]) * (rr - r[2])
			+ f[s + IY + IZ /*     */] * (rr - r[0]) * (r[1] - ll) * (r[2] - ll)
			+ f[s + IY /*           */] * (rr - r[0]) * (r[1] - ll)
					* (rr - r[2])
			+ f[s + IZ /*          */] * (rr - r[0]) * (rr - r[1]) * (r[2] - ll)
			+ f[s /*               */] * (rr - r[0]) * (rr - r[1])
					* (rr - r[2]);
}

CUDA_DEVICE
void cache_scatter(Real f[CACHE_SIZE], Real v, Real const *r0, Real const *r1) {
	Real r[3] = { r0[0] - r1[0], r0[1] - r1[1], r0[2] - r1[2] };
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

CUDA_DEVICE Real id_to_shift_[][3] = { //
		{ 0, 0, 0 },           // 000
				{ _R, 0, 0 },           // 001
				{ 0, _R, 0 },           // 010
				{ 0, 0, _R },          // 011
				{ _R, _R, 0 },           // 100
				{ _R, 0, _R },          // 101
				{ 0, _R, _R },          // 110
				{ 0, _R, _R },          // 111
		};
CUDA_DEVICE int sub_index_to_id_[4][3] = { //
		{ 0, 0, 0 }, /*VERTEX*/
		{ 1, 2, 4 }, /*EDGE*/
		{ 6, 5, 3 }, /*FACE*/
		{ 7, 7, 7 } /*VOLUME*/

		};

#undef _R

/* @formatter:on*/

CUDA_GLOBAL void spBorisYeePush_kernel(struct spPage *pg, Real cmr, double dt,
		const Real *E, const Real *B, const Real *inv_dx) {
	int i = threadIdx.x;
	CUDA_SHARED Real tE[3][CACHE_SIZE], tB[3][CACHE_SIZE];

	while (pg != 0x0) {
		if (pg->tag & (0x1 << i) != 0) {

			struct boris_point_s *p = (struct boris_point_s *) (pg->data
					+ i * sizeof(struct boris_point_s));
			Real E[3], B[3];

			cache_gather(&E[0], tE[0], p->r,
					id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
			cache_gather(&E[1], tE[1], p->r,
					id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
			cache_gather(&E[2], tE[2], p->r,
					id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);

			cache_gather(&B[0], tB[0], p->r,
					id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
			cache_gather(&B[1], tB[1], p->r,
					id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
			cache_gather(&B[2], tB[2], p->r,
					id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

			spBorisPushOne(p, cmr, dt, E, B, inv_dx);

		}
		pg = pg->next;
	}

}

CUDA_HOST void spBorisYeePush(struct spPage *pg, Real cmr, double dt,
		const Real *E, const Real *B, size_type const *i_self_,
		size_type const *i_lower_, size_type const *i_upper_,
		const Real *inv_dx) {

//    size_t els_size_in_byte = sizeof(struct boris_point_s);

	/* @formatter:off*/
	int numBlocks = 1;
	dim3 threadsPerBlock(SP_NUMBER_OF_ELEMENT_IN_PAGE, 1);
	spBorisYeePush_kernel<<<numBlocks, threadsPerBlock>>>(pg, cmr, dt, tE, tB,
			inv_dx);
	/* @formatter:on*/

}
#define DEFINE_INTEGRAL_FUNC(_FUN_PREFIX_, _TYPE_, _ATOMIC_FUN_)                                               \
CUDA_GLOBAL void                                                                                               \
_FUN_PREFIX_##_kernel(struct spPage *pg, Real *cf, size_type iform, size_type sub_index)                       \
{                                                                                                              \
    for (int i = 0, ie = SP_NUMBER_OF_ELEMENT_IN_PAGE; i < ie; ++i)                                            \
    {                                                                                                          \
        struct _TYPE_ *p = (struct _TYPE_ *) (pg->data + i * sizeof(struct _TYPE_));                           \
        cache_scatter(&cf[threadIdx.x + (threadIdx.y + threadIdx.z * CACHE_EXTENT_Y)*CACHE_EXTENT_X],          \
                    _ATOMIC_FUN_(p, sub_index), p->r,                                                          \
                      id_to_shift_[sub_index_to_id_[iform][sub_index]]);                                       \
    }                                                                                                          \
}                                                                                                              \
CUDA_HOST void                                                                                                 \
_FUN_PREFIX_(struct spPage *pg, Real tf[CACHE_SIZE], size_type iform, size_type sub_index)                     \
{                                                                                                              \
    while (pg != 0x0)                                                                                          \
    {                                                                                                          \
        int numBlocks = 1;                                                                                     \
        dim3 threadsPerBlock(CACHE_EXTENT_X,CACHE_EXTENT_Y,CACHE_EXTENT_Z);                                    \
        _FUN_PREFIX_##_kernel<<<numBlocks,threadsPerBlock >>> (pg, tf,iform,sub_index);                        \
        pg = pg->next;                                                                                         \
    }                                                                                                          \
}                                                                                                              \


DEFINE_INTEGRAL_FUNC(spBorisYeeIntegralRho, boris_point_s, spBorisGetRho)

DEFINE_INTEGRAL_FUNC(spBorisYeeIntegralJ, boris_point_s, spBorisGetJ)

DEFINE_INTEGRAL_FUNC(spBorisYeeIntegralE, boris_point_s, spBorisGetE)

