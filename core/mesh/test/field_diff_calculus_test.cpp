/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "field_diff_calculus_test.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(TestFieldCase, TestFETL,

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
