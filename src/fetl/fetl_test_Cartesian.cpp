/*
 * fetl_test.cpp
 *
 *  Created on: 2013年12月28日
 *      Author: salmon
 */

#include "../mesh/mesh_rectangle.h"
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"

#define TMESH Mesh<CartesianGeometry<UniformArray>>

#include "fetl.h"
#include "fetl_test.h"
//#include "fetl_test1.h"
#include "fetl_test2.h"
//#include "fetl_test3.h"
//#include "fetl_test4.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(FETLCartesian, TestFETL,

testing::Combine(testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0 })  //
        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })

),

testing::Values(

nTuple<3, Real>( { 5.0, 2.0, 3.0 })  //
        , nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //

        ),

testing::Values(

nTuple<3, size_t>( { 1, 1, 100 }) //
        , nTuple<3, size_t>( { 100, 1, 1 }) //
        , nTuple<3, size_t>( { 1, 100, 1 }) //
        , nTuple<3, size_t>( { 1, 1, 10 }) //
        , nTuple<3, size_t>( { 1, 10, 20 }) //
        , nTuple<3, size_t>( { 51, 1, 51 }) //
        , nTuple<3, size_t>( { 51, 51, 1 }) //
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
