/**
 *  @file field_diff_calculus_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <simpla/algebra/all.h>
#include <simpla/physics/Constants.h>
#include "field_diff_calculus_test.h"

using namespace simpla;

#ifndef CYLINDRICAL_COORDINATE_SYSTEM

INSTANTIATE_TEST_CASE_P(
        DiffCalculusCartesian, FETLTest,

        testing::Combine(

                testing::Values(

//                        std::make_tuple(nTuple<Real, 3>{{0, 0, 0}}, nTuple<Real, 3>({1.0, 2.0, 3.0})), //
                        std::make_tuple(nTuple<Real, 3>({0.0, -2.0, -3.0}), nTuple<Real, 3>({1.0, 2.0, 3.0}))


                ),


                testing::Values(
                        nTuple<size_type, 3>{1, 1, 1},     //
                        nTuple<size_type, 3>{10, 1, 1}, //
                        nTuple<size_type, 3>{1, 100, 1},//
                        nTuple<size_type, 3>{1, 1, 100},//
                        nTuple<size_type, 3>{1, 10, 20},//
                        nTuple<size_type, 3>{17, 1, 17},//
                        nTuple<size_type, 3>{10, 1, 1},//
                        nTuple<size_type, 3>{5, 19, 17} //
                ),

                testing::Values(nTuple<Real, 3>{TWOPI, 3 * TWOPI, TWOPI})

        ));

#else
INSTANTIATE_TEST_CASE_P(
        DiffCalculusCylindrical, FETLTest,

        testing::Combine(

                testing::Values(

                        std::make_tuple(nTuple<Real, 3>({1.0, 0, 0}),
                                         nTuple<Real, 3>({2.0, 2.0, TWOPI})),

                        std::make_tuple(nTuple<Real, 3>({10.0, -2.0, 0.0}),
                                       nTuple<Real, 3>({12.0, 1.0, HALFPI}))

                ),


                testing::Values(
                        nTuple<size_t, 3>({16, 20, 16})
                        , nTuple<size_t, 3>({16, 1, 1}) //
                        , nTuple<size_t, 3>({40, 1, 50}) //
                        , nTuple<size_t, 3>({17, 20, 1}) //
//                        , nTuple<size_t, 3>({1, 10, 1}) //
//                        , nTuple<size_t, 3>({1, 1, 16}) //
//                        , nTuple<size_t, 3>({1, 20, 20}) //
                             //
                ),

                testing::Values(nTuple<Real, 3>({TWOPI, TWOPI, TWOPI}))

        ));
#endif