//
// Created by salmon on 17-1-13.
//

#include <gtest/gtest.h"
#include "simpla/data/DataType.h"
using namespace simpla::data;
TEST(DataTest, data_type) {
    auto dtype = DataType::create<double>();
    std::cout << dtype.name() << std::endl;
}
