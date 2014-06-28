/*
 * geometry_cartesian_test.cpp
 *
 *  Created on: 2014年6月27日
 *      Author: salmon
 */

#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"

#define GEOMETRY  CartesianGeometry<UniformArray,false>
#include "geometry_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(Test_CartesianGeometry, TestGeometry,

testing::Combine(

testing::Values(nTuple<3, Real>( { 1.0, 0.0, 0.0, }) //
        , nTuple<3, Real>( { 1.0, -2.0, -3.0 })    //
//        , nTuple<3, Real>( { 1.0, 1.0, 1.0 })    //
        ),

testing::Values(
//
        nTuple<3, Real>( { 2.0, 0.0, 0.0 }), //
        nTuple<3, Real>( { 0.0, 2.0, 0.0 }), //
        nTuple<3, Real>( { 0.0, 0.0, 2.0 }), //
        nTuple<3, Real>( { 0.0, 2.0, 2.0 }), //
        nTuple<3, Real>( { 2.0, 0.0, 2.0 }), //
        nTuple<3, Real>( { 2.0, 2.0, 0.0 }), //
        nTuple<3, Real>( { 1.0, 3.0, 2.0 })  //

                ),

testing::Values(

nTuple<3, size_t>( { 10, 1, 1 }) //
        , nTuple<3, size_t>( { 11, 1, 1 }) //
        , nTuple<3, size_t>( { 1, 17, 1 }) //
        , nTuple<3, size_t>( { 1, 1, 10 }) //
        , nTuple<3, size_t>( { 1, 17, 17 }) //
        , nTuple<3, size_t>( { 17, 1, 17 }) //
        , nTuple<3, size_t>( { 17, 17, 1 }) //
        , nTuple<3, size_t>( { 13, 16, 10 })   //

        )));
