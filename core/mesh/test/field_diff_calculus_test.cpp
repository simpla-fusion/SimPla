/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include "field_diff_calculus_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(DiffCalculus, FETLTest,

testing::Combine(

testing::Values(nTuple<Real, 3>( { 0.0, 0.0, 0.0 })  //
		, nTuple<Real, 3>( { -1.0, -2.0, -3.0 })

),

testing::Values(nTuple<Real, 3>( { 1.0, 0.0, 0.0 })  //
		, nTuple<Real, 3>( { 0.0, 1.0, 0.0 }) //
		, nTuple<Real, 3>( { 0.0, 0.0, 2.0 }) //
		, nTuple<Real, 3>( { 0.0, 2.0, 2.0 }) //
		, nTuple<Real, 3>( { 2.0, 0.0, 2.0 }) //
		, nTuple<Real, 3>( { 2.0, 2.0, 2.0 }) //
		),

testing::Values( //nTuple<size_t, 3>( { 1, 1, 1 }) //
		nTuple<size_t, 3>( { 100, 1, 1 }) //
				, nTuple<size_t, 3>( { 1, 200, 1 }) //
				, nTuple<size_t, 3>( { 1, 1, 10 }) //
				, nTuple<size_t, 3>( { 1, 10, 20 }) //
				, nTuple<size_t, 3>( { 17, 1, 17 }) //
				, nTuple<size_t, 3>( { 17, 17, 1 }) //
				, nTuple<size_t, 3>( { 12, 16, 10 })   //
				),

testing::Values(nTuple<Real, 3>( { TWOPI, 3 * TWOPI, TWOPI }))

));
