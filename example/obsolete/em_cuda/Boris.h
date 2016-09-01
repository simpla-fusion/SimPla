/**
 * Boris.h
 *
 *  Created on: 2016-6-15
 *      Author: salmon
 */

#ifndef BORIS_H_
#define BORIS_H_
#include "../../sp_lite/sp_def.h"
#include "../../sp_lite/spParticle.h" //for POINT_HEAD
#include "../../sp_lite/spMesh.h" //for POINT_HEAD

#define boris_buffer_s_DEPTH SP_NUMBER_OF_ENTITIES_IN_PAGE * 2

struct boris_buffer_s
{
	int tag[boris_buffer_s_DEPTH];
	Real r[3][boris_buffer_s_DEPTH];
	Real v[3][boris_buffer_s_DEPTH];
	Real f[boris_buffer_s_DEPTH];
	Real w[boris_buffer_s_DEPTH];
};

struct boris_point_s
{
	POINT_HEAD
	Real v[3];
	Real f;
	Real w;
};

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
//
//#define _R 1.0
//MC_CONSTANT Real id_to_shift_[][3] =
//{ //
//		{ 0, 0, 0 },           // 000
//				{ _R, 0, 0 },           // 001
//				{ 0, _R, 0 },           // 010
//				{ 0, 0, _R },          // 011
//				{ _R, _R, 0 },           // 100
//				{ _R, 0, _R },          // 101
//				{ 0, _R, _R },          // 110
//				{ 0, _R, _R },          // 111
//		};
//MC_CONSTANT int sub_index_to_id_[4][3] =
//{ //
//		{ 0, 0, 0 }, /*VERTEX*/
//		{ 1, 2, 4 }, /*EDGE*/
//		{ 6, 5, 3 }, /*FACE*/
//		{ 7, 7, 7 } /*VOLUME*/
//
//		};
//MC_CONSTANT int cache_cell_offset_tag[CACHE_SIZE] =
//{ };
//MC_CONSTANT size_type cache_cell_offset[CACHE_SIZE] =
//{ };
//MC_HOST_DEVICE_PREFIX
//inline void spBorisPushOne(struct boris_point_s const *p0, struct boris_point_s *p1, Real dt, Real q, Real m,
//		Real const tE[3][CACHE_SIZE], Real const tB[3][CACHE_SIZE], Real tJ[4][CACHE_SIZE], const Real *invD)
//{
//
//	Real E[3], B[3];
//
//	cache_gather(&E[0], tE[0], p0->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
//	cache_gather(&E[1], tE[1], p0->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
//	cache_gather(&E[2], tE[2], p0->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
//
//	cache_gather(&B[0], tB[0], p0->r, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
//	cache_gather(&B[1], tB[1], p0->r, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
//	cache_gather(&B[2], tB[2], p0->r, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);
//
//	p1->r[0] = p0->r[0] + p0->v[0] * dt * 0.5 * invD[0];
//	p1->r[1] = p0->r[1] + p0->v[1] * dt * 0.5 * invD[1];
//	p1->r[2] = p0->r[2] + p0->v[2] * dt * 0.5 * invD[2];
//
//	Real v_[3], t[3];
//
//	t[0] = B[0] * (q / m * dt * 0.5);
//	t[1] = B[1] * (q / m * dt * 0.5);
//	t[2] = B[2] * (q / m * dt * 0.5);
//
//	p1->v[0] = p0->v[0] + E[0] * (q / m * dt * 0.5);
//	p1->v[1] = p0->v[1] + E[1] * (q / m * dt * 0.5);
//	p1->v[2] = p0->v[2] + E[2] * (q / m * dt * 0.5);
//
//	v_[0] = p1->v[0] + (p1->v[1] * t[2] - p1->v[2] * t[1]);
//	v_[1] = p1->v[1] + (p1->v[2] * t[0] - p1->v[0] * t[2]);
//	v_[2] = p1->v[2] + (p1->v[0] * t[1] - p1->v[1] * t[0]);
//
//	Real tt = t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + 1.0;
//
//	p1->v[0] += (v_[1] * t[2] - v_[2] * t[1]) * 2.0 / tt;
//	p1->v[1] += (v_[2] * t[0] - v_[0] * t[2]) * 2.0 / tt;
//	p1->v[2] += (v_[0] * t[1] - v_[1] * t[0]) * 2.0 / tt;
//
//	p1->v[0] += E[0] * (q / m * dt * 0.5);
//	p1->v[1] += E[1] * (q / m * dt * 0.5);
//	p1->v[2] += E[2] * (q / m * dt * 0.5);
//
//	p1->r[0] += p1->v[0] * dt * 0.5 * invD[0];
//	p1->r[1] += p1->v[1] * dt * 0.5 * invD[1];
//	p1->r[2] += p1->v[2] * dt * 0.5 * invD[2];
//
//	cache_scatter(tJ[0], p1->f * p1->w * q, p1->r, id_to_shift_[sub_index_to_id_[0/*VERTEX*/][0]]);
//	cache_scatter(tJ[1], p1->f * p1->w * p1->v[0] * q, p1->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
//	cache_scatter(tJ[2], p1->f * p1->w * p1->v[1] * q, p1->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
//	cache_scatter(tJ[3], p1->f * p1->w * p1->v[2] * q, p1->r, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
//
//}
#endif /* BORIS_H_ */
