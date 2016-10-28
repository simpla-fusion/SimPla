//
// Created by salmon on 16-10-8.
//



#include <gtest/gtest.h>
#include <iostream>

#include "simpla/toolbox/MemoryDataBase.h"

using namespace simpla::toolbox;

TEST(DataBaseMemory, general)
{
    DataBaseMemory db;

    db.set("a", 1.0);
    db.set("b", 2);
    db.set("c", "hello world!");


    EXPECT_DOUBLE_EQ(db.template as<double>("a"), 1.0);
    EXPECT_EQ(db.template as<int>("b"), 2);
    EXPECT_EQ(db.template as<std::string>("c"), std::string("hello world!"));

    auto db2 = std::make_shared<DataBaseMemory>();
    db2->set("h", "this is the second");

    auto foo = db.create("d");

    foo->set("e", 3);
    foo->set("f", 34.5);
    foo->set("g", db2);

    std::cout << db << std::endl;
    db.set("a", "hello world!");
    db.set("b", "hello world!");
    std::cout << db << std::endl;
}