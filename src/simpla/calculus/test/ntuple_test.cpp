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
#include "simpla/toolbox/type_traits.h"
#include "simpla/calculus/nTuple.h"
#include "simpla/calculus/Calculus.h"

//#include "../primitives.h"
//#include "../mpl.h"
#include "simpla/toolbox/integer_sequence.h"

using namespace simpla;
using namespace simpla::calculus;

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
//
//        traits::seq_for_each(extents(),
//
//                             [&](size_t const idx[traits::extent<extents>::value])
//                             {
//                                 traits::index(aA, idx) = idx[0] * 2;
//                                 traits::index(aB, idx) = 5 - idx[0];
//                                 traits::index(aC, idx) = idx[0] * 5 + 1;
//                                 traits::index(aD, idx) = 0;
//                                 traits::index(vA, idx) = traits::index(aA, idx);
//                                 traits::index(vB, idx) = traits::index(aB, idx);
//                                 traits::index(vC, idx) = traits::index(aC, idx);
//                                 traits::index(vD, idx) = 0;
//
//                                 traits::index(res, idx) = -(traits::index(aA, idx) + a) /
//                                                           (traits::index(aB, idx) * b - c) - traits::index(aC, idx);
//
//                             });

        num_of_loops = 1000000L;
    }

public:

    std::size_t num_of_loops = 10000000L;

    typedef T type;

//    typedef traits::extents<type> extents;

    nTuple<std::size_t, calculus::traits::rank<type>::value> DIMENSIONS;

    typedef calculus::traits::value_type_t<type> value_type;

    type vA, vB, vC, vD;

//    typename traits::pod_type<T>::type aA, aB, aC, aD, res;

    value_type a, b, c, d;

};

typedef testing::Types<
        nTuple<double, 3>
//        Matrix<double, 3, 3>,
//        Tensor<double, 3, 4, 5>,
//        Tensor<int, 3, 4, 5, 6>,
//        Tensor<std::complex<double>, 3, 4, 5, 6>

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
//                                     traits::index(TestFixture::aA, idx) - traits::index(TestFixture::vB, idx)));
//                             EXPECT_DOUBLE_EQ(0, std::abs(
//                                     traits::index(TestFixture::aB, idx) - traits::index(TestFixture::vA, idx)));
//                         });
//
//
//}
//
//TYPED_TEST(TestNtuple, assign_Scalar)
//{
//
//
//    TestFixture::vA = TestFixture::a;
//
//    traits::seq_for_each(typename TestFixture::extents(),
//                         [&](size_t const idx[traits::extent<typename TestFixture::extents, 0>::value])
//                         {
//                             EXPECT_DOUBLE_EQ(0, abs(TestFixture::a - traits::index(TestFixture::vA, idx)));
//                         }
//    );
//
//
//}
//
//TYPED_TEST(TestNtuple, assign_Array)
//{
//
//    TestFixture::vA = TestFixture::aA;
//
//    traits::seq_for_each(typename TestFixture::extents(),
//                         [&](size_t const idx[traits::extent<typename TestFixture::extents, 0>::value])
//                         {
//                             EXPECT_DOUBLE_EQ(0, abs(traits::index(TestFixture::aA, idx) -
//                                                     traits::index(TestFixture::vA, idx)));
//                         }
//    );
//
//
//}
//
//TYPED_TEST(TestNtuple, self_assign)
//{
//
//    TestFixture::vB += TestFixture::vA;
//
//    traits::seq_for_each(typename TestFixture::extents(),
//                         [&](size_t const idx[traits::extent<typename TestFixture::extents, 0>::value])
//                         {
//                             EXPECT_DOUBLE_EQ(0, abs(traits::index(TestFixture::vB, idx)
//                                                     - (traits::index(TestFixture::aB, idx) +
//                                                        traits::index(TestFixture::aA, idx))));
//
//                         }
//    );
//
//
//}
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

    TestFixture::vD = TestFixture::vA + TestFixture::vB;//EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC);

//    traits::seq_for_each(typename TestFixture::extents(),
//                         [&](size_t const idx[traits::extent<typename TestFixture::extents, 0>::value])
//                         {
//                             auto ta = traits::index(TestFixture::vA, idx);
//                             auto tb = traits::index(TestFixture::vB, idx);
//                             auto tc = traits::index(TestFixture::vC, idx);
//                             auto td = traits::index(TestFixture::vD, idx);
//
//                             EXPECT_DOUBLE_EQ(0, abs(EQUATION(ta, tb, tc) - td));
//                         }
//    );


}
