/*
 * topology_test_UniformArray.cpp
 *
 *  Created on: 2014年6月27日
 *      Author: salmon
 */
#define TOPOLOGY UniformArray

#include "uniform_array.h"

#include "topology_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(SimPla, TestTopology, testing::Values(

nTuple<3, size_t>( { 10, 1, 1 }) //
        , nTuple<3, size_t>( { 11, 1, 1 }) //
        , nTuple<3, size_t>( { 1, 17, 1 }) //
        , nTuple<3, size_t>( { 1, 1, 10 }) //
        , nTuple<3, size_t>( { 1, 17, 17 }) //
        , nTuple<3, size_t>( { 17, 1, 17 }) //
        , nTuple<3, size_t>( { 17, 17, 1 }) //
        , nTuple<3, size_t>( { 13, 16, 10 })   //

        ));
