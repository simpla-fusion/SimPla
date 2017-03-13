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

    LOGGER << "Registered DataBackend: " << SingletonHolder<DataBackendFactory>::instance().RegisteredBackend()
           << std::endl;

    DataTable db;

    db.Set("CartesianGeometry", "hello world!");
    db.Set("d", {1, 2, 3, 4, 5, 56, 6, 6});
    db.Set("g", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    db.Set("e", {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}});
    db.Set({"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
            "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
    db.Set("h", {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}});
    db.Set("i", {"abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
    db.Set("j", {"abc"_ = {"abc"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
    db.Set("b.a", 5);
    //    db.Set("/b/sub/1/2/3/4/d/123456", nTuple<int, 3>{1, 2, 3});
    db.Set("/b/sub/e", nTuple<int, 4>{1, 2, 3, 4});
    db.Add("/b/sub/c", nTuple<int, 4>{5, 6, 7, 8});
    db.Add("/b/sub/c", nTuple<int, 4>{1, 5, 3, 4});
    db.Add("/b/sub/c", nTuple<int, 4>{2, 5, 3, 4});
    db.Add("/b/sub/c", nTuple<int, 4>{3, 5, 3, 4});
    db.Add("/b/sub/c", nTuple<int, 4>{4, 5, 3, 4});
    db.Add("/b/sub/d", {1, 2});
    db.Add("/b/sub/d", 5);
    db.Add("/b/sub/d", 5);
    db.Add("/b/sub/d", 5);

    //    db.Add("/b/sub/c", "la la land");
    db.Add("/b/sub/a", {3, 5, 3, 4});
    db.Add("/b/sub/a", 9);

    LOGGER << "a =" << (db.Get("a")->as<bool>(false)) << std::endl;

    LOGGER << "/b/sub/e  = " << db.Get("/b/sub/e")->as<nTuple<int, 4>>() << std::endl;
    LOGGER << "db: " << db << std::endl;

}

TEST(DataTable, lua) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << SingletonHolder<DataBackendFactory>::instance().RegisteredBackend()
           << std::endl;
    DataTable lua_db(std::string("lua:///home/salmon/workspace/SimPla/test/data/test.lua"));
    LOGGER << "lua:// " << *lua_db.Get("Context") << std::endl;
}

TEST(DataTable, samrai) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << SingletonHolder<DataBackendFactory>::instance().RegisteredBackend()
           << std::endl;
    DataTable samrai_db("samrai://ggg/b/b/c/d/a?{{123,4},{123,}}");
    samrai_db.Set("/d", {1, 2, 3, 4, 5, 56, 6, 6});
    samrai_db.Set("/d/e/f", "Just atest");
    samrai_db.Set("/d/e/g", {"a"_ = "Just a test", "b"_ = 1235.5});
    samrai_db.Set("/d/e/e", 1.23456);

    LOGGER << *samrai_db.backend() << std::endl;
}