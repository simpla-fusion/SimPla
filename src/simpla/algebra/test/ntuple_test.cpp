/**
 * @file ntuple_test.cpp
 *
 *  created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include "../nTuple.h"
#include "../Expression.h"
#include "../Arithmetic.h"
#include "../Algebra.h"

using namespace simpla;

#define EQUATION(_A, _B, _C)  ( -(_A  +TestFixture::a )/(   _B *TestFixture::b -TestFixture::c  )- _C)

template<typename T>
class TestNtuple : public testing::Test
{
protected:

    virtual void SetUp()
    {

        a = 1;
        b = 3;
        c = 4;
        d = 7;

//        DIMENSIONS = extents();

        traits::seq_for_each(extents(),
                             [&](size_t const *idx)
                             {
                                 algebra::get_v(aA, idx) = static_cast<value_type>(idx[0] * 2);
                                 algebra::get_v(aB, idx) = static_cast<value_type>(5 - idx[0]);
                                 algebra::get_v(aC, idx) = static_cast<value_type>(idx[0] * 5 + 1);
                                 algebra::get_v(aD, idx) = static_cast<value_type>(0);
                                 algebra::get_v(vA, idx) = algebra::get_v(aA, idx);
                                 algebra::get_v(vB, idx) = algebra::get_v(aB, idx);
                                 algebra::get_v(vC, idx) = algebra::get_v(aC, idx);
                                 algebra::get_v(vD, idx) = static_cast<value_type>(0);
                                 algebra::get_v(res, idx) = -(algebra::get_v(aA, idx) + a) /
                                                            (algebra::get_v(aB, idx) * b - c) -
                                                            algebra::get_v(aC, idx);

                             });

        num_of_loops = 1000000L;
    }

public:

    std::size_t num_of_loops = 10000000L;

    typedef T type;

    typedef algebra::traits::extents<type> extents;

    nTuple<size_type, algebra::traits::rank<type>::value> DIMENSIONS;

    typedef algebra::traits::value_type_t<type> value_type;

    type vA, vB, vC, vD;

    typename algebra::traits::pod_type_t<type> aA, aB, aC, aD, res;

    value_type a, b, c, d;

};

typedef testing::Types<
        nTuple<double, 3>//,
        , nTuple<std::complex<double>, 3> //
        , Matrix<double, 3, 3> //
        , Tensor<double, 3, 4, 5>//
        , Tensor<int, 3, 4, 5, 6> //
        , Tensor<std::complex<double>, 3, 4, 5, 6>

> ntuple_type_lists;

TYPED_TEST_CASE(TestNtuple, ntuple_type_lists);
//
//TYPED_TEST(TestNtuple, swap)
//{
//    std::swap(TestFixture::vA, TestFixture::vB);
//
//    traits::seq_for_each(typename TestFixture::extents(),
//                         [&](size_t const idx[traits::extent<typename TestFixture::extents, 0>::value])
//                         {
//                             EXPECT_DOUBLE_EQ(0, std::abs(
//                                     algebra::get_v(TestFixture::aA, idx) - algebra::get_v(TestFixture::vB, idx)));
//                             EXPECT_DOUBLE_EQ(0, std::abs(
//                                     algebra::get_v(TestFixture::aB, idx) - algebra::get_v(TestFixture::vA, idx)));
//                         });
//
//
//}
//
TYPED_TEST(TestNtuple, assign_Scalar)
{

    TestFixture::vA = TestFixture::a;

    traits::seq_for_each(
            typename TestFixture::extents(),
            [&](size_type const *idx)
            {
                EXPECT_DOUBLE_EQ(0, abs(TestFixture::a - algebra::get_v(TestFixture::vA, idx)));
            }
    );


}

TYPED_TEST(TestNtuple, assign_Array)
{

    TestFixture::vA = TestFixture::aA;

    traits::seq_for_each(typename TestFixture::extents(),
                         [&](size_type const *idx)
                         {
                             EXPECT_DOUBLE_EQ(0, abs(algebra::get_v(TestFixture::aA, idx) -
                                                     algebra::get_v(TestFixture::vA, idx)));
                         }
    );


}

TYPED_TEST(TestNtuple, self_assign)
{

    TestFixture::vB += TestFixture::vA;

    traits::seq_for_each(
            typename TestFixture::extents(),
            [&](size_type const *idx)
            {
                EXPECT_DOUBLE_EQ(0, abs(
                        (algebra::get_v(TestFixture::vB, idx)) -
                        (algebra::get_v(TestFixture::aA, idx)) -
                        (algebra::get_v(TestFixture::aB, idx)))
                );

            }
    );


}
////
//TYPED_TEST(TestNtuple, cross)
//{
//
//    nTuple<typename TestFixture::value_type, 3> vA, vB, vC, vD;
//
//    for (int i = 0; i < 3; ++i)
//    {
//        vA[i] = (i * 2);
//        vB[i] = (5 - i);
//    }
//
//    for (int i = 0; i < 3; ++i)
//    {
//        vD[i] = vA[(i + 1) % 3] * vB[(i + 2) % 3]
//                - vA[(i + 2) % 3] * vB[(i + 1) % 3];
//    }
//
//    vC = cross(vA, vB);
//    vD -= vC;
//    EXPECT_DOUBLE_EQ(0, abs(vD[0]) + abs(vD[1]) + abs(vD[2]));
//
//}

TYPED_TEST(TestNtuple, arithmetic)
{

    TestFixture::vD = EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC);
    //
    traits::seq_for_each(typename TestFixture::extents(),
                         [&](size_type const *idx)
                         {
                             auto ta = algebra::get_v(TestFixture::vA, idx);
                             auto tb = algebra::get_v(TestFixture::vB, idx);
                             auto tc = algebra::get_v(TestFixture::vC, idx);
                             auto td = algebra::get_v(TestFixture::vD, idx);
                             EXPECT_DOUBLE_EQ(0, abs(EQUATION(ta, tb, tc) - td));
                         }
    );


}
