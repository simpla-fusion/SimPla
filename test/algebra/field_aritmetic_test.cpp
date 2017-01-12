//
// Created by salmon on 17-1-12.
//

#include <gtest/gtest.h>
#include <simpla/algebra/all.h>

using namespace simpla;
using namespace algebra;
namespace sat = simpla::algebra::traits;
struct DummyMesh {};

// TEST(TestAlgebra, iform_arithmetic) {
//    EXPECT_EQ((sat::iform<double>::value), VERTEX);
//    EXPECT_EQ((sat::iform<int>::value), VERTEX);
//    EXPECT_EQ((sat::iform<nTuple<double, 3>>::value), VERTEX);
//    EXPECT_EQ((sat::iform<nTuple<int, 3>>::value), VERTEX);
//}
template <typename TPara>
class TestAlgebra : public testing::Test {};

template <typename TM, typename TV, int IFORM, int DOF>
struct TestAlgebra<Field<TM, TV, IFORM, DOF>> : public testing::Test {
    static constexpr int iform = IFORM;
    static constexpr int dof = DOF;

    typedef TV value_type;
    typedef Field<TM, value_type, IFORM, DOF> field_type;
    typedef Field<TM, value_type, VERTEX, 1> scalar_field_type;
};

template <typename TM, typename TV, int IFORM, int DOF>
constexpr int TestAlgebra<Field<TM, TV, IFORM, DOF>>::iform;
template <typename TM, typename TV, int IFORM, int DOF>
constexpr int TestAlgebra<Field<TM, TV, IFORM, DOF>>::dof;

typedef testing::Types<

    Field<DummyMesh, Real, EDGE>,       //
    Field<DummyMesh, Real, FACE>,       //
    Field<DummyMesh, Real, VOLUME>,     //
    Field<DummyMesh, Real, VERTEX, 3>,  //
    Field<DummyMesh, Real, EDGE, 3>,    //
    Field<DummyMesh, Real, FACE, 3>,    //
    Field<DummyMesh, Real, VOLUME, 3>   //
    >
    TypeParamList;

TYPED_TEST_CASE(TestAlgebra, TypeParamList);

TYPED_TEST(TestAlgebra, iform_base) {
    EXPECT_EQ((sat::iform<typename TestFixture::field_type>::value), TestFixture::iform);
    EXPECT_EQ((sat::dof<typename TestFixture::field_type>::value), TestFixture::dof);
}

TYPED_TEST(TestAlgebra, iform_arithmetic) {
    typedef typename TestFixture::field_type field_type;
    typedef typename TestFixture::scalar_field_type scalar_field_type;

    EXPECT_EQ(sat::iform<decltype(std::declval<field_type>() * 2.0)>::value, TestFixture::iform);
    EXPECT_EQ(sat::iform<decltype(std::declval<field_type>() / 2.0)>::value, TestFixture::iform);
    EXPECT_EQ(sat::iform<decltype(-std::declval<field_type>())>::value, TestFixture::iform);

    EXPECT_EQ(
        sat::iform<decltype(-std::declval<field_type>() * 2 - std::declval<field_type>())>::value,
        TestFixture::iform);

    EXPECT_EQ(
        sat::iform<decltype(std::declval<field_type>() * std::declval<scalar_field_type>())>::value,
        TestFixture::iform);
    EXPECT_EQ(
        sat::iform<decltype(std::declval<field_type>() / std::declval<scalar_field_type>())>::value,
        TestFixture::iform);
    EXPECT_EQ(sat::iform<decltype(std::declval<field_type>() + std::declval<field_type>())>::value,
              TestFixture::iform);
    EXPECT_EQ(sat::iform<decltype(std::declval<field_type>() - std::declval<field_type>())>::value,
              TestFixture::iform);
}
// TEST(AlgebraTest, iform_calculus) {
//    EXPECT_EQ(sat::iform<decltype(grad(-std::declval<rho>() * 2))>::value, EDGE);
//    EXPECT_EQ(sat::iform<decltype(grad(-std::declval<vrho>() * 2))>::value, FACE);
//
//    EXPECT_EQ(sat::iform<decltype(curl(-std::declval<E>() * 2) * 3.0)>::value, FACE);
//    EXPECT_EQ(sat::iform<decltype(curl(-std::declval<B>() * 2) * 3.0)>::value, EDGE);
//
//    EXPECT_EQ(sat::iform<decltype(diverge(-std::declval<E>() * 2))>::value, VERTEX);
//    EXPECT_EQ(sat::iform<decltype(diverge(-std::declval<B>() * 2))>::value, VOLUME);
//}