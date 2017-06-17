//
// Created by salmon on 17-1-12.
//

#include <gtest/gtest.h>
#include <simpla/algebra/all.h>

using namespace simpla;
using namespace algebra;

struct DummyMesh {
    typedef size_type entity_id;
    typedef Real scalar_type;
};

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
    //
    Field<DummyMesh, Real, VERTEX>,     //
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
//    EXPECT_EQ(sat::GetIFORM<decltype(grad(-std::declval<n>() * 2))>::value, EDGE);
//    EXPECT_EQ(sat::GetIFORM<decltype(grad(-std::declval<vrho>() * 2))>::value, FACE);
//
//    EXPECT_EQ(sat::GetIFORM<decltype(curl(-std::declval<E>() * 2) * 3.0)>::value, FACE);
//    EXPECT_EQ(sat::GetIFORM<decltype(curl(-std::declval<B>() * 2) * 3.0)>::value, EDGE);
//
//    EXPECT_EQ(sat::GetIFORM<decltype(diverge(-std::declval<E>() * 2))>::value, VERTEX);
//    EXPECT_EQ(sat::GetIFORM<decltype(diverge(-std::declval<B>() * 2))>::value, VOLUME);
//}

TEST(TestAlgebra0, iform_not_field) {
    EXPECT_EQ((sat::iform<double>::value), VERTEX);
    EXPECT_EQ((sat::iform<int>::value), VERTEX);
    EXPECT_EQ((sat::iform<nTuple<double, 3>>::value), VERTEX);
    EXPECT_EQ((sat::iform<nTuple<int, 3>>::value), VERTEX);
}
typedef Field<DummyMesh, Real, VERTEX, 3> f0t3;

typedef Field<DummyMesh, Real, VERTEX> f0t;
typedef Field<DummyMesh, Real, EDGE> f1t;
typedef Field<DummyMesh, Real, FACE> f2t;
typedef Field<DummyMesh, Real, VOLUME> f3t;

namespace simpla {
namespace algebra {
namespace traits {
template <typename T>
struct iform<declare::Expression<tags::negate, T>> : public int_const<iform<T>::value> {};
}
}
}
TEST(TestAlgebra0, iform_vector_calculus) {
    EXPECT_EQ((sat::iform<decltype(grad(std::declval<f0t>()))>::value), EDGE);
    EXPECT_EQ((sat::iform<decltype(diverge(std::declval<f1t>()))>::value), VERTEX);
    EXPECT_EQ((sat::iform<decltype(curl(std::declval<f1t>()))>::value), FACE);
    EXPECT_EQ((sat::iform<decltype(curl(std::declval<f2t>()))>::value), EDGE);

    EXPECT_EQ((sat::iform<decltype(-std::declval<f1t>() * 0.2)>::value), EDGE);
    EXPECT_EQ((sat::iform<decltype((std::declval<f1t>() * 0.2) / 4.0)>::value), EDGE);
    EXPECT_EQ(
        (sat::iform<
            declare::Expression<algebra::tags::multiplies, Real,
                                declare::Expression<algebra::tags::multiplies, f1t, Real>>>::value),
        EDGE);

    EXPECT_EQ((sat::iform<decltype(hodge_star(std::declval<f0t>()))>::value), VOLUME);
    EXPECT_EQ((sat::iform<decltype(hodge_star(std::declval<f1t>()))>::value), FACE);
    EXPECT_EQ((sat::iform<decltype(hodge_star(std::declval<f2t>()))>::value), EDGE);
    EXPECT_EQ((sat::iform<decltype(hodge_star(std::declval<f3t>()))>::value), VERTEX);
    EXPECT_EQ((sat::iform<decltype(wedge(std::declval<f0t>(), std::declval<f3t>()))>::value),
              VOLUME);

    EXPECT_EQ((sat::iform<decltype(wedge(std::declval<f1t>(), std::declval<f1t>()))>::value), FACE);

    EXPECT_EQ((sat::iform<decltype(wedge(std::declval<f1t>(), std::declval<f2t>()))>::value),
              VOLUME);
//    EXPECT_EQ(
//        (sat::GetIFORM<decltype(inner_product(std::declval<f1t>(), std::declval<f1t>()))>::value),
//        VERTEX);

    EXPECT_EQ(
        (sat::iform<decltype(-inner_product(std::declval<f2t>(), std::declval<f2t>()) * 2)>::value),
        VOLUME);

    EXPECT_EQ((sat::iform<decltype(-grad(std::declval<f0t>()) * 2)>::value), EDGE);
    EXPECT_EQ((sat::iform<decltype(-diverge(std::declval<f1t>()))>::value), VERTEX);
    EXPECT_EQ((sat::iform<decltype(-curl(std::declval<f1t>()))>::value), FACE);
    EXPECT_EQ((sat::iform<decltype(-curl(std::declval<f2t>()))>::value), EDGE);

    EXPECT_EQ((sat::iform<decltype(diverge(std::declval<f0t3>()))>::value), VERTEX);
    EXPECT_EQ((sat::iform<decltype(curl(std::declval<f0t3>()))>::value), VERTEX);
    EXPECT_EQ((sat::iform<decltype(curl(std::declval<f0t3>()))>::value), VERTEX);
}