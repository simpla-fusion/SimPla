//
// Created by salmon on 17-7-2.
//

#include <gtest/gtest.h>

#include <complex>
#include <iostream>
#include <typeinfo>
#include "simpla/algebra/Array.h"
#include "simpla/algebra/sfc/z_sfc.h"
using namespace simpla;

#define EQUATION(_A, _B, _C) (-(_A + TestFixture::a) / (_B * TestFixture::b - TestFixture::c) - _C)

template <typename T>
class TestArray : public testing::Test {
   protected:
    virtual void SetUp() {}

   public:
    typedef T type;

    index_box_type idx_box = {{1, 4, 7}, {2, 6, 9}};

    nTuple<int, std::rank<type>::value> DIMENSIONS;

    typedef traits::value_type_t<type> value_type;

    type vA{idx_box}, vB{idx_box}, vC{idx_box}, vD{idx_box};

    value_type a = 1, b = 2, c = 3, d = 4;
};

typedef testing::Types<Array<double>,               //
                       Array<int>,                  //
                       Array<std::complex<double>>  //
                       >
    array_type_lists;

TYPED_TEST_CASE(TestArray, array_type_lists);

#define ARRAY_FOREACH                                                                                                 \
    for (index_type i = std::get<0>(TestFixture::idx_box)[0], ie = std::get<1>(TestFixture::idx_box)[0]; i < ie; ++i) \
        for (index_type j = std::get<0>(TestFixture::idx_box)[1], je = std::get<1>(TestFixture::idx_box)[1]; j < je;  \
             ++j)                                                                                                     \
            for (index_type k = std::get<0>(TestFixture::idx_box)[2], ke = std::get<1>(TestFixture::idx_box)[2];      \
                 k < ke; ++k)

TYPED_TEST(TestArray, assign_Scalar) {
    TestFixture::vA = TestFixture::a;
    ARRAY_FOREACH { EXPECT_DOUBLE_EQ(0, std::abs(TestFixture::vA(i, j, k) - TestFixture::a)); }
}
//
// TYPED_TEST(TestArray, swap) {
//    TestFixture::vA.swap(TestFixture::vB);
//
//    ARRAY_FOREACH {
//        EXPECT_DOUBLE_EQ(0, simpla::abs(TestFixture::aA(i, j, k) - TestFixture::vB(i, j, k)));
//        EXPECT_DOUBLE_EQ(0, simpla::abs(TestFixture::aB(i, j, k) - TestFixture::vA(i, j, k)));
//    }
//}
//
////

TYPED_TEST(TestArray, clear) {
    TestFixture::vB.Clear();
    ARRAY_FOREACH { EXPECT_DOUBLE_EQ(0, simpla::abs(TestFixture::vB(i, j, k))); }
}
TYPED_TEST(TestArray, assign_Array) {
    TestFixture::vA = [&](index_type i, index_type j, index_type k) { return i + j + k; };

    ARRAY_FOREACH {
        EXPECT_DOUBLE_EQ(0,
                         std::abs(TestFixture::vA(i, j, k) - static_cast<typename TestFixture::value_type>(i + j + k)));
    }
}

TYPED_TEST(TestArray, self_assign) {
    TestFixture::vA = [&](index_type i, index_type j, index_type k) { return i + j + k; };
    TestFixture::vB.Clear();

    TestFixture::vB += TestFixture::vA;

    ARRAY_FOREACH {
        EXPECT_DOUBLE_EQ(0,
                         std::abs(TestFixture::vB(i, j, k) - static_cast<typename TestFixture::value_type>(i + j + k)));
    }
    TestFixture::vB += TestFixture::vA;
    ARRAY_FOREACH {
        EXPECT_DOUBLE_EQ(
            0, std::abs(TestFixture::vB(i, j, k) - static_cast<typename TestFixture::value_type>(2 * (i + j + k))));
    }
}
//
// TYPED_TEST(TestArray, cross) {
//    nTuple<typename TestFixture::value_type, 3> vA, vB, vC, vD;
//
//    for (int i = 0; i < 3; ++i) {
//        vA[i] = (i * 2);
//        vB[i] = (5 - i);
//    }
//
//    for (int i = 0; i < 3; ++i) { vD[i] = vA[(i + 1) % 3] * vB[(i + 2) % 3] - vA[(i + 2) % 3] * vB[(i + 1) % 3]; }
//
//    vC = cross(vA, vB);
//    vD -= vC;
//    EXPECT_DOUBLE_EQ(0, abs(vD[0]) + abs(vD[1]) + abs(vD[2]));
//}
//
// TYPED_TEST(TestArray, dot) {
//    typename TestFixture::value_type expect;
//    expect = 0;
//
//    ARRAY_FOREACH { expect += TestFixture::aA[i][j][k] * TestFixture::aB[i][j][k]; }
//
//    EXPECT_DOUBLE_EQ(0.0, abs(expect - (dot(TestFixture::vA, TestFixture::vB))));
//}
//
TYPED_TEST(TestArray, arithmetic) {
    TestFixture::vA = [&](index_type i, index_type j, index_type k) { return i + j + k; };
    TestFixture::vB = [&](index_type i, index_type j, index_type k) { return i + 2 * j + k; };
    TestFixture::vC = [&](index_type i, index_type j, index_type k) { return i + j + 2 * k; };
    TestFixture::vD = EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC);

    ARRAY_FOREACH {
        auto ta = TestFixture::vA(i, j, k);
        auto tb = TestFixture::vB(i, j, k);
        auto tc = TestFixture::vC(i, j, k);
        auto td = TestFixture::vD(i, j, k);

        EXPECT_DOUBLE_EQ(0, abs(td - EQUATION(ta, tb, tc)));
    }
}

// TYPED_TEST(TestArray, compare) {
//    TestFixture::vA = [&](index_type i, index_type j, index_type k) { return i + j + k; };
//    TestFixture::vB = [&](index_type i, index_type j, index_type k) { return i + 2 * j + k; };
//    TestFixture::vC = [&](index_type i, index_type j, index_type k) { return i + j + 2 * k; };
//
//    EXPECT_TRUE(TestFixture::vA == TestFixture::vA);
//    EXPECT_FALSE(TestFixture::vA != TestFixture::vA);
//    EXPECT_FALSE(TestFixture::vA == TestFixture::vB);
//    EXPECT_TRUE(TestFixture::vA != TestFixture::vB);
//}