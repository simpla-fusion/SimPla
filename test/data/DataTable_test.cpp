//
// Created by salmon on 17-1-6.
//

#include <gtest/gtest.h>
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <simpla/toolbox/FancyStream.h>
#include <complex>
#include <iostream>

using namespace simpla;
using namespace simpla::data;
TEST(DataTable, memory) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;

    DataTable db;

    db.SetValue("CartesianGeometry", "hello world!");
    db.SetValue("d", {1, 2, 3, 4, 5, 56, 6, 6});
    db.SetValue("g", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    db.SetValue("e", {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}});
    db.SetValue({"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
                 "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
    db.SetValue("h", {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}});
    db.SetValue("i", {"abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
    db.SetValue("j", {"abc"_ = {"abc"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
    db.SetValue("b.a", 5);
    db.SetValue("/b/sub/1/2/3/4/d/123456", nTuple<int, 3>{1, 2, 3});
    db.SetValue("/b/sub/e", nTuple<int, 4>{1, 2, 3, 4});
    db.AddValue("/b/sub/c", nTuple<int, 4>{5, 6, 7, 8});
    db.AddValue("/b/sub/c", nTuple<int, 4>{1, 5, 3, 4});
    db.AddValue("/b/sub/c", nTuple<int, 4>{2, 5, 3, 4});
    db.AddValue("/b/sub/c", nTuple<int, 4>{3, 5, 3, 4});
    db.AddValue("/b/sub/c", nTuple<int, 4>{4, 5, 3, 4});
    db.AddValue("/b/sub/d", {1, 2});
    db.AddValue("/b/sub/d", 5);
    db.AddValue("/b/sub/d", 5);
    db.AddValue("/b/sub/d", 5);

    db.AddValue("/b/sub/a", {3, 5, 3, 4});
    db.AddValue("/b/sub/a", 9);
    LOGGER << "db: " << db << std::endl;

    LOGGER << "a =" << (db.GetValue<bool>("a", false)) << std::endl;
    LOGGER << "/b/sub/e  = " << db.GetValue<nTuple<int, 4>>("/b/sub/e") << std::endl;
}

TEST(DataTable, lua) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
    DataTable lua_db(std::string("/home/salmon/workspace/SimPla/test/data/test.lua"));
    LOGGER << "lua:// " << *lua_db.Get("Context") << std::endl;
}

TEST(DataTable, samrai) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
    DataTable samrai_db("samrai://");
    samrai_db.SetValue("f", {1, 2, 3, 4, 5, 56, 6, 6});
    samrai_db.SetValue("/d/e/f", "Just atest");
    samrai_db.SetValue("/d/e/g", {"a"_ = "la la land", "b"_ = 1235.5});
    samrai_db.SetValue("/d/e/e", 1.23456);

    LOGGER << *samrai_db.backend() << std::endl;
}

TEST(DataTable, hdf5) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
    DataTable db("test.h5", "w");
    db.SetValue("pi", 3.1415926);
    db.SetValue("a", "just a test");
//    db.SetValue("c", {1.2346, 4.0, 5.0, 6.0, 6.1});
    //    db.SetValue({"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
    //                 "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
    //    db.SetValue("h", {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}});
    db.SetValue("i", {"abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
    db.SetValue("j", {"abc"_ = {"abc"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
    //    db.AddValue("/b/sub/d", {1, 2});
    //    db.AddValue("/b/sub/d", 5);
    //    db.AddValue("/b/sub/d", 5);
    //    db.AddValue("/b/sub/d", 5);

    //    db.SetValue("/b/sub/a", {3, 5, 3, 4});
    db.SetValue("/b/sub/b", 9);
    LOGGER << "h5:// " << db << std::endl;
}