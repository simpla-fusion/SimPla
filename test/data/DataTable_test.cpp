//
// Created by salmon on 17-1-6.
//

#include <gtest/gtest.h>
#include <complex>
#include <iostream>
#include "simpla/SIMPLA_config.h"
#include "simpla/data/Data.h"
#include "simpla/utilities/FancyStream.h"
#include "simpla/utilities/SingletonHolder.h"

using namespace simpla;
using namespace simpla::data;
TEST(DataTable, memory) {
    auto db = DataNode::New();

    *(*db)["CartesianGeometry"] = {"hello world!"};
    *(*db)["d"] = {1, 2, 3, 4, 5, 56, 6, 6};
    (*db)["g"]->SetValue<nTuple<int, 2, 2, 2>>({{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    //    *(*db)["strlist"] = {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}};
    *(*db)["b/a"] = (5);

    *(*db)["/b/sub/1/2/3/4/d/123456"] = {1, 2, 3};

    *(*db)["/b/sub/c"] += {5, 6, 7, 8};
    *(*db)["/b/sub/c"] += {1, 5, 3, 4};
    *(*db)["/b/sub/c"] += {2, 5, 3, 4};
    *(*db)["/b/sub/c"] += {3, 5, 3, 4};
    *(*db)["/b/sub/c"] += {4, 5, 3, 4};

    //    db->AddValue("/b/sub/d", 1);
    //    db->AddValue("/b/sub/d", 5);
    //    db->AddValue("/b/sub/d", 5);
    //    db->AddValue("/b/sub/d", 5);
    //    //
    *(*db)["/b/sub/d"] += "wa wa";
    *(*db)["/b/sub/d"] += "la la";

    *(*db)["/b/sub/a"] += {3, 5, 3, 4};
    *(*db)["/b/sub/a"] += {3, 5, 3, 4};

    *(*db)["/b/sub/e"] = {1, 2, 3, 4};
    *(*db)["/b/sub/e"] += 9;

    *(*db)["a"] += {"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
                    "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}};
    //    *(*db)["h"] = {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}};
    *(*db)["i"] = {"default"_, "abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"};
    *(*db)["i"] += {"abc"_ = {"abc1"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}};
    LOGGER << db->GetNumberOfChildren() << std::endl;
    LOGGER << "db: " << (*db) << std::endl;
    EXPECT_EQ(db->GetNode("b/a")->as<int>(), 5);

    EXPECT_EQ((*db)["/CartesianGeometry"]->as<std::string>(), "hello world!");
}

TEST(DataTable, lua) {
    auto db = DataNode::New("lua://");
    db->Parse(
        "PI = 3.141592653589793\n"
        "Context = {\n"
        "    c = 299792458.0, -- m/s\n"
        "    qe = 1.60217656e-19, -- C\n"
        "    me = 9.10938291e-31, --kg\n"
        "    mp = 1.672621777e-27, --kg\n"
        "    mp_me = 1836.15267245, --\n"
        "    KeV = 1.1604e7, -- K\n"
        "    Tesla = 1.0, -- Tesla\n"
        "    TWOPI = PI * 2,\n"
        "    k_B = 1.3806488e-23, --Boltzmann_constant\n"
        "    epsilon0 = 8.8542e-12,\n"
        "    AAA = { c =  3 , d = { c = \"3\", e = { 1, 3, 4, 5 } } },\n"
        "    CCC = { 1, 3, 4, 5 }\n"
        "}");
    LOGGER << "lua:// " << *(*db)["/Context"] << std::endl;
    EXPECT_EQ((*db)["/Context/AAA/c"]->as<int>(), 3);
    //    EXPECT_EQ(((*db)["/Context/CCC"]->as<nTuple<int, 4>>()), (nTuple<int, 4>{1, 3, 4, 5}));

    EXPECT_DOUBLE_EQ((*db)["/Context/c"]->as<double>(), 299792458);

    //   db->Set("box", {{1, 2, 3}, {4, 5, 6}});
    //    LOGGER << "box  = " <<db->Get<std::tuple<nTuple<int, 3>, nTuple<int, 3>>>("box") << std::endl;
}
//
// TEST(DataTable, samrai) {
//    logger::set_stdout_level(1000);
//
//    LOGGER << "Registered DataBase: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
//    DataTable db("samrai://");
//    //   db->Set("f", {1, 2, 3, 4, 5, 56, 6, 6});
//    //   db->Set("/d/e/f", "Just atest");
//    //   db->Set("/d/e/g", {"a"_ = "la la land", "b"_ = 1235.5});
//    //   db->Set("/d/e/e", 1.23456);
//   db->Set("box", {{1, 2, 3}, {4, 5, 6}});
//    LOGGER << *db.database() << std::endl;
//    LOGGER << "box  = " <<db->Get<std::tuple<nTuple<int, 3>, nTuple<int, 3>>>("box") << std::endl;
//
//}
//
// TEST(DataTable, hdf5) {
//    logger::set_stdout_level(1000);
//
//    //    LOGGER << "Registered DataBase: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
//    auto db = DataBase::New("test.h5");
//    db->Set("pi", 3.1415926);
//    db->Set("a", "just a test");
//    //   db->Set("c", {1.2346, 4.0, 5.0, 6.0, 6.1});
//    //   db->Set({"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
//    //                 "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !",
//    "hello!"}});
//    //   db->Set("h", {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}});
//    db->Set("i", {"abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
//    db->Set("j", {"abc"_ = {"abc"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
//    //   db->AddValue("/b/sub/d", {1, 2});
//    //   db->AddValue("/b/sub/d", 5);
//    //   db->AddValue("/b/sub/d", 5);
//    //   db->AddValue("/b/sub/d", 5);
//
//    //   db->Set("/b/sub/a", {3, 5, 3, 4});
//    db->Set("/b/sub/b", 9);
//    LOGGER << "h5:// " << db << std::endl;
//    db->Set("box", {{1, 2, 3}, {4, 5, 6}});
//    LOGGER << "box  = " << db->Get<std::tuple<nTuple<int, 3>, nTuple<int, 3>>>("box") << std::endl;
//}