/*
 * mesh_test.cpp
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include "mesh_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(SimPla, TestMesh,

testing::Combine(

testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, }) //
//        , nTuple<3, Real>( { -1.0, -1.0, -1.0 })    //
//        , nTuple<3, Real>( { 1.0, 1.0, 1.0 })    //
        ),

testing::Values(

//nTuple<3, Real>( { 2.0, 0.0, 0.0 }), //
//nTuple<3, Real>( { 0.0, 2.0, 0.0 }), //
//nTuple<3, Real>( { 0.0, 0.0, 2.0 }), //
//nTuple<3, Real>( { 0.0, 2.0, 2.0 }), //
//nTuple<3, Real>( { 2.0, 0.0, 2.0 }), //
//nTuple<3, Real>( { 2.0, 2.0, 0.0 }), //
nTuple<3, Real>( { 1.0, 2.0, 2.0 })  //

        ),

testing::Values(

nTuple<3, size_t>( { 10, 10, 1  }) //
//        , nTuple<3, size_t>( { 17, 1, 1 }) //
//        , nTuple<3, size_t>( { 1, 17, 1 }) //
//        , nTuple<3, size_t>( { 1, 1, 10 }) //
//        , nTuple<3, size_t>( { 1, 17, 17 }) //
//        , nTuple<3, size_t>( { 17, 1, 17 }) //
//        , nTuple<3, size_t>( { 17, 17, 1 }) //
//        , nTuple<3, size_t>( { 13, 16, 10 })   //

        )));
