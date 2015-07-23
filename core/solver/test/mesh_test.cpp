/**
 * @file mesh_test.cpp
 *
 * @date 2015-5-7
 * @author salmon
 */

#include "mesh_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(RectMesh, MeshTest,

testing::Combine(testing::Values(0, 1, 6),

testing::Values(nTuple<Real, 3>( { 0.0, 0.0, 0.0 })  //
		, nTuple<Real, 3>( { -1.0, -2.0, -3.0 })

),

testing::Values(nTuple<Real, 3>( { 1.0, 2.0, 3.0 })  //
		, nTuple<Real, 3>( { 0.0, 1.0, 0.0 }) //
		, nTuple<Real, 3>( { 0.0, 0.0, 2.0 }) //
		, nTuple<Real, 3>( { 0.0, 2.0, 2.0 }) //
		, nTuple<Real, 3>( { 2.0, 0.0, 2.0 }) //
		, nTuple<Real, 3>( { 2.0, 2.0, 2.0 }) //
		),

testing::Values(nTuple<size_t, 3>( { 1, 1, 1 }) //
		, nTuple<size_t, 3>( { 100, 1, 1 }) //
		, nTuple<size_t, 3>( { 1, 200, 1 }) //
		, nTuple<size_t, 3>( { 1, 1, 10 }) //
		, nTuple<size_t, 3>( { 3, 4, 1 }) //
//				, nTuple<size_t, 3>( { 17, 1, 17 }) //
//				, nTuple<size_t, 3>( { 17, 17, 1 }) //
//				, nTuple<size_t, 3>( { 12, 16, 10 })   //
		)

		));
