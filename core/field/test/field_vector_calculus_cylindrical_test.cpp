/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../../manifold/fetl.h"

using namespace simpla;


typedef Manifold<CylindricalCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> m_type;
typedef Real v_type;

#include "field_vector_calculus_test.h"

//INSTANTIATE_TEST_CASE_P(FETLCartesian, FETLTest,
//
//testing::Combine(testing::Values(
//
//nTuple<Real, 3>( { 0.0, 0.0, 0.0 }), nTuple<Real, 3>( { -1.0, -2.0, -3.0 })
//
//),
//
//testing::Values(
//
//nTuple<Real, 3>( { 1.0, 2.0, 1.0 }) //
//
//		, nTuple<Real, 3>( { 2.0, 0.0, 0.0 }) //
//		, nTuple<Real, 3>( { 0.0, 2.0, 0.0 }) //
//		, nTuple<Real, 3>( { 0.0, 0.0, 2.0 }) //
//		, nTuple<Real, 3>( { 0.0, 2.0, 2.0 }) //
//		, nTuple<Real, 3>( { 2.0, 0.0, 2.0 }) //
//		, nTuple<Real, 3>( { 2.0, 2.0, 0.0 }) //
//
//		),
//
//testing::Values(
//
//nTuple<size_t, 3>( { 40, 12, 10 }) //
//		, nTuple<size_t, 3>( { 100, 1, 1 }) //
//		, nTuple<size_t, 3>( { 1, 100, 1 }) //
//		, nTuple<size_t, 3>( { 1, 1, 100 }) //
//		, nTuple<size_t, 3>( { 1, 10, 5 }) //
//		, nTuple<size_t, 3>( { 11, 1, 21 }) //
//		, nTuple<size_t, 3>( { 11, 21, 1 }) //
//		),
//
//testing::Values(
//
//nTuple<Real, 3>( { TWOPI, 3 * TWOPI, TWOPI }))
//
//));

