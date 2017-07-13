//
// Created by salmon on 17-2-15.
//

#include <gtest/gtest.h"
#include "simpla/engine/Attribute.h"

using namespace simpla;

TEST(TestAttribute, GUID) {
    engine::Attribute f("E", typeid(Real), FACE, 1);
    engine::Attribute f0("E", typeid(Real), FACE, 1);
    engine::Attribute f1("F", typeid(Real), FACE, 1);
    engine::Attribute f2("E", typeid(Real), FACE, 2);
    engine::Attribute f3("E", typeid(Real), EDGE, 1);
    engine::Attribute f4("E", typeid(int), FACE, 1);

    EXPECT_EQ(f.GetGUID(), f0.GetGUID());
    EXPECT_NE(f.GetGUID(), f1.GetGUID());
    EXPECT_NE(f.GetGUID(), f2.GetGUID());
    EXPECT_NE(f.GetGUID(), f3.GetGUID());
    EXPECT_NE(f.GetGUID(), f4.GetGUID());
}
struct DummyMesh {};
template <typename V, typename M, int IFORM, int DOF>
struct DummyField {
    typedef V value_type;
    typedef M mesh_type;
    static constexpr int iform = IFORM;
    static constexpr int dof = DOF;
    DummyField() {}
    ~DummyField() {}
    virtual void Initialize(){};
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const {};
};
template <typename V, typename M, int IFORM, int DOF>
using TAttribute = engine::AttributeViewAdapter<DummyField<V, M, IFORM, DOF>>;
TEST(TestAttribute, GUID_Adapter) {
    TAttribute<Real, DummyMesh, FACE, 1> f{"E"};
    TAttribute<Real, DummyMesh, FACE, 1> f0{"E"};
    TAttribute<Real, DummyMesh, FACE, 1> f1{"F"};
    TAttribute<Real, DummyMesh, FACE, 2> f2{"E"};
    TAttribute<Real, DummyMesh, EDGE, 1> f3{"E"};
    TAttribute<int, DummyMesh, FACE, 1> f4{"E"};

    EXPECT_EQ(std::type_index(typeid(DummyMesh)).hash_code(), f.mesh_type_index().hash_code());
    EXPECT_EQ(f.GetGUID(), f0.GetGUID());
    EXPECT_NE(f.GetGUID(), f1.GetGUID());
    EXPECT_NE(f.GetGUID(), f2.GetGUID());
    EXPECT_NE(f.GetGUID(), f3.GetGUID());
    EXPECT_NE(f.GetGUID(), f4.GetGUID());
}
