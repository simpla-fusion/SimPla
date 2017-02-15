//
// Created by salmon on 17-2-15.
//

#include <gtest/gtest.h>
#include <simpla/engine/AttributeView.h>

using namespace simpla;
template <typename V, int IFORM, int DOF>
struct TAttribute : public engine::AttributeView {
    TAttribute(std::string const& s) : engine::AttributeView(s, typeid(V), IFORM, DOF) {}
    ~TAttribute() {}
};
TEST(TestAttributeView, GUID) {
    TAttribute<Real, FACE, 1> f{"E"};
    TAttribute<Real, FACE, 1> f0{"E"};
    TAttribute<Real, FACE, 1> f1{"F"};
    TAttribute<Real, FACE, 2> f2{"E"};
    TAttribute<Real, EDGE, 1> f3{"E"};
    TAttribute<int, FACE, 1> f4{"E"};

    EXPECT_EQ(f.GUID(), f0.GUID());
    EXPECT_NE(f.GUID(), f1.GUID());
    EXPECT_NE(f.GUID(), f2.GUID());
    EXPECT_NE(f.GUID(), f3.GUID());
    EXPECT_NE(f.GUID(), f4.GUID());
}
