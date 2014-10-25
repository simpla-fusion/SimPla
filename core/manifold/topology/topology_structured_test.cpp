/*
 * @file topology_structured_test.cpp
 *
 * @date created on: 2014-6-27
 *      @Author: salmon
 */

#include "structured.h"

#define TOPOLOGY StructuredMesh

#include "topology_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(Test_Structured, TestTopology,
		testing::Values(nTuple<size_t, 3>( { 10, 1, 1 }) //
				, nTuple<size_t, 3>( { 1, 17, 1 }) //
				, nTuple<size_t, 3>( { 1, 1, 10 }) //
				, nTuple<size_t, 3>( { 1, 17, 17 }) //
				, nTuple<size_t, 3>( { 17, 1, 17 }) //
				, nTuple<size_t, 3>( { 17, 17, 1 }) //
				, nTuple<size_t, 3>( { 13, 16, 10 }   //
				)));
