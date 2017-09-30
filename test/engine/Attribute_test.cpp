//
// Created by salmon on 17-2-15.
//

#include "simpla/engine/Attribute.h"
#include <gtest/gtest.h>
#include "simpla/data/Data.h"
using namespace simpla;
using namespace simpla::data;

TEST(TestAttribute, GUID) {
    engine::AttributeT<Real, NODE> f;
    engine::AttributeT<Real, NODE, 3> g;

    EXPECT_EQ(f.GetRank(), 0);
    EXPECT_EQ(g.GetRank(), 1);

    std::cout << f[0] << std::endl;
    std::cout << g[0] << std::endl;
}
