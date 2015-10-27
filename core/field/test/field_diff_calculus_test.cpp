/**
 *  @file field_diff_calculus_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include "field_diff_calculus_test.h"

using namespace simpla;

#ifdef CYLINDRICAL_COORDINATE_SYTEM

INSTANTIATE_TEST_CASE_P(DiffCalculus, FETLTest,

        testing::Combine(

                testing::Values(nTuple<Real, 3>(
                                {	1.0, 0.0, 0.0})),

                testing::Values(

//nTuple<Real, 3>( { 2.0, 0.0, 0.0 })  //
//		, nTuple<Real, 3>( { 2.0, 1.0, 0.0 }) //
//		, nTuple<Real, 3>( { 2.0, 0.0, 2.0 * TWOPI }) //
//		, nTuple<Real, 3>( { 2.0, 2.0, 2.0 * TWOPI }) //
//		, nTuple<Real, 3>( { 2.0, 0.0, 2.0 * TWOPI }) //,
                        nTuple<Real, 3>(
                                {	2.0, 2.0, 2.0 * TWOPI}) //
                ),

                testing::Values(//nTuple<size_t, 3>( { 1, 1, 1 }) //
//		nTuple<size_t, 3>( { 100, 1, 1 }) //
//				, nTuple<size_t, 3>( { 1, 200, 1 }) //
//				, nTuple<size_t, 3>( { 1, 1, 10 }) //
//				, nTuple<size_t, 3>( { 1, 10, 20 }) //
//				, nTuple<size_t, 3>( { 17, 1, 17 }) //
//				, nTuple<size_t, 3>( { 17, 17, 1 }) //,
                        nTuple<size_t, 3>(
                                {	12, 16, 10})   //
                ),

                testing::Values(nTuple<Real, 3>(
                                {	TWOPI, 3 * TWOPI, TWOPI}))

        ));

#else

INSTANTIATE_TEST_CASE_P(DiffCalculus, FETLTest,

                        testing::Combine(

                                testing::Values(

                                        std::make_tuple(nTuple<Real, 3>({0, 0, 0}),
                                                        nTuple<Real, 3>({1.0, 2.0, 3.0})),

                                        std::make_tuple(nTuple<Real, 3>({-1.0, -2.0, -3.0}),
                                                        nTuple<Real, 3>({1.0, 1.0, 1.0}))


                                ),


                                testing::Values(
                                        nTuple<size_t, 3>({100, 1, 1}) //
                                        , nTuple<size_t, 3>({1, 200, 1}) //
                                        , nTuple<size_t, 3>({1, 1, 100}) //
                                        , nTuple<size_t, 3>({1, 10, 20}) //
                                        , nTuple<size_t, 3>({17, 1, 17}) //
                                        , nTuple<size_t, 3>({10, 20, 1}) //
                                        , nTuple<size_t, 3>({5, 19, 17})   //
                                ),

                                testing::Values(nTuple<Real, 3>({TWOPI, 3 * TWOPI, TWOPI}))

                        ));
#endif
