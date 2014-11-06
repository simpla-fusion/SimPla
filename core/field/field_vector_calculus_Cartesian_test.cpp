/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../utilities/ntuple.h"
#include "../manifold/manifold.h"
#include "../manifold/geometry/cartesian.h"
#include "../manifold/topology/structured.h"
#include "../manifold/diff_scheme/fdm.h"
#include "../manifold/interpolator/interpolator.h"
#include "field.h"

using namespace simpla;
//
//int main(int argc, char **argv)
//{
//	typedef Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
//			FiniteDiffMethod, InterpolatorLinear> manifold_type;
//
//	std::shared_ptr<manifold_type> manifold = std::make_shared<manifold_type>();
//
//	auto f3 = make_form<Real, VOLUME>(manifold);
//	auto f1 = make_form<Real, EDGE>(manifold);
//	auto f2 = make_form<Real, FACE>(manifold);
//	auto f0 = make_form<Real, VERTEX>(manifold);
//
//	f1 = codifferential_derivative(-exterior_derivative(f1));
//}

#define TMESH Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,FiniteDiffMethod, InterpolatorLinear>

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

