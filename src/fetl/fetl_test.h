/*
 * fetl_test.h
 *
 *  Created on: 2014年2月20日
 *      Author: salmon
 */

#ifndef FETL_TEST_H_
#define FETL_TEST_H_
#include <gtest/gtest.h>
#include "fetl.h"
#include "fetl_test_suit.h"
#include "fetl_test1.h"
#include "fetl_test2.h"
#include "fetl_test3.h"
#include "fetl_test4.h"

using namespace simpla;

INSTANTIATE_TEST_CASE_P(FETLEuclidean, TestFETL,

testing::Combine(testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })  //
        , nTuple<3, Real>( { -1.0, -2.0, -3.0 })

),

testing::Values(

nTuple<3, Real>( { 1.0, 2.0, 3.0 })  //
        , nTuple<3, Real>( { 2.0, 0.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 0.0 }) //
        , nTuple<3, Real>( { 0.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 0.0, 2.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //

        ),

testing::Values(

nTuple<3, size_t>( { 1, 1, 1 }) //
        , nTuple<3, size_t>( { 17, 1, 1 }) //
        , nTuple<3, size_t>( { 1, 17, 1 }) //
        , nTuple<3, size_t>( { 1, 1, 10 }) //
        , nTuple<3, size_t>( { 1, 10, 20 }) //
        , nTuple<3, size_t>( { 17, 1, 17 }) //
        , nTuple<3, size_t>( { 17, 17, 1 }) //
        , nTuple<3, size_t>( { 12, 16, 10 })   //

        )

        ));

#endif /* FETL_TEST_H_ */
