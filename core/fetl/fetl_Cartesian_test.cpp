/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */


#include "../manifold/manifold.h"
#include "../manifold/geometry/cartesian.h"
#include "../manifold/topology/structured.h"
using namespace simpla;
#define TMESH   Manifold<CartesianCoordinates<StructuredMesh>>


#include "fetl.h"
#include "fetl_test.h"
#include "fetl_test1.h"
//#include "fetl_test2.h"
//#include "fetl_test3.h"
//#include "fetl_test4.h"

INSTANTIATE_TEST_CASE_P(FETLCartesian, TestFETL,

testing::Combine(testing::Values(

		nTuple<3, Real>( { 0.0, 0.0, 0.0 })  //
//        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })

),

testing::Values(

nTuple<3, Real>( { 5.0, 2.0, 3.0 })

//, nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
//        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
//        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
//        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
//        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
//        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //

        ),

testing::Values(

nTuple<3, size_t>( { 1, 1, 100 }) //

//        , nTuple<3, size_t>( { 100, 1, 1 }) //
//        , nTuple<3, size_t>( { 1, 100, 1 }) //
//        , nTuple<3, size_t>( { 1, 1, 100 }) //
//        , nTuple<3, size_t>( { 1, 100, 50 }) //
//        , nTuple<3, size_t>( { 51, 1, 51 }) //
//        , nTuple<3, size_t>( { 51, 51, 1 }) //
        ),

testing::Values(nTuple<3, Real>( { TWOPI, 3 * TWOPI, TWOPI }))

));

//INSTANTIATE_TEST_CASE_P(FETLEuclideanPerformance, TestFETL,
//
//testing::Combine(
//
//testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })),
//
//testing::Values(nTuple<3, Real>( { 5.0, 2.0, 3.0 })),
//
//testing::Values(nTuple<3, size_t>( { 124, 134, 154 }))
//
//));
