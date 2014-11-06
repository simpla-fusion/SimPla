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

using namespace simpla;

#define TMESH Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,FiniteDiffMethod, InterpolatorLinear>

#define VEC_VALUE_TYPE nTuple<Real,3>

#include "field_vector_calculus_test.h"


//INSTANTIATE_TEST_CASE_P(FETLEuclideanPerformance, TestFETL,
//
//testing::Combine(
//
//testing::Values(nTuple<Real,3>( { 0.0, 0.0, 0.0, })),
//
//testing::Values(nTuple<Real,3>( { 5.0, 2.0, 3.0 })),
//
//testing::Values(nTuple<size_t,3>( { 124, 134, 154 }))
//
//));
