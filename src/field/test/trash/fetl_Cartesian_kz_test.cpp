/*
 * fetl_test_Euclidean_kz.cpp
 *
 *  created on: 2014-6-23
 *      Author: salmon
 */

#include "../mesh/mesh_rectangle.h"
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"

using namespace simpla;
#define TMESH Mesh<CartesianGeometry<UniformArray>,true >

#include "fetl_test.h"
#include "fetl_test1.h"
#include "fetl_test2.h"
#include "fetl_test3.h"
#include "fetl_test4.h"


INSTANTIATE_TEST_CASE_P(FETLCartesianKz, TestFETL,

testing::Combine(testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })  //
        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })

),

testing::Values(nTuple<3, Real>( { 1.0, 0.0, 0.0 })  //
        , nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 2.0, 2.0 }) //
        ),

testing::Values(
//		nTuple<3, size_t>( { 1, 1, 1 }) //,
        nTuple<3, size_t>( { 100, 1, 1 }) //
                , nTuple<3, size_t>( { 1, 17, 1 }) //
                , nTuple<3, size_t>( { 1, 1, 10 }) //
                , nTuple<3, size_t>( { 1, 10, 20 }) //
                , nTuple<3, size_t>( { 17, 1, 17 }) //
                , nTuple<3, size_t>( { 17, 17, 1 }) //
                , nTuple<3, size_t>( { 12, 16, 10 })   //
                ),

testing::Values(nTuple<3, Real>( { TWOPI, 3 * TWOPI, TWOPI }))

));
