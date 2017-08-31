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
// TEST(DataTable, lua) {
//    auto db = DataNode::New("lua://");
//    db->Parse(
//        "PI = 3.141592653589793  "
//        "_ROOT_={\n"
//        "c = 299792458.0, -- m/s\n"
//        "qe = 1.60217656e-19, -- C\n"
//        "me = 9.10938291e-31, --kg\n"
//        "mp = 1.672621777e-27, --kg\n"
//        "mp_me = 1836.15267245, --\n"
//        "KeV = 1.1604e7, -- K\n"
//        "Tesla = 1.0, -- Tesla\n"
//        "TWOPI =  PI * 2,\n"
//        "k_B = 1.3806488e-23, --Boltzmann_constant\n"
//        "epsilon0 = 8.8542e-12,\n"
//        "AAA = { c =  3 , d = { c = \"3\", e = { 1, 3, 4, 5 } } },\n"
//        "CCC = { 1, 3, 4, 5 },\n"
//        "Box={{1,2,3},{3,4,5}} \n"
//        "}");
//    MESSAGE << "lua:// " << (*db) << std::endl;
//    //    MESSAGE << "Box " << (*db)["Context/Box"]->as<nTuple<int, 2, 3>>() << std::endl;
//    EXPECT_EQ((*db)["Context/AAA/c"]->as<int>(), 3);
//    EXPECT_EQ(((*db)["/Context/CCC"]->as<nTuple<int, 4>>()), (nTuple<int, 4>{1, 3, 4, 5}));
//    //
//    //    EXPECT_DOUBLE_EQ((*db)["/Context/c"]->as<double>(), 299792458);
//
//    //   db->SetEntity("box", {{1, 2, 3}, {4, 5, 6}});
//    //    LOGGER << "box  = " <<db->GetEntity<std::tuple<nTuple<int, 3>, nTuple<int, 3>>>("box") << std::endl;
//}

class DataBaseTest : public testing::TestWithParam<std::string> {
   protected:
    void SetUp() {
        logger::set_stdout_level(logger::LOG_VERBOSE);
        m_url = GetParam();
        db = DataNode::New(m_url);
        //        MESSAGE << " Data URL : \"" << m_url << "\"" << std::endl;
    }
    void TearDown() {}

   public:
    virtual ~DataBaseTest() {}
    std::shared_ptr<DataNode> db = nullptr;
    std::string m_url;
};

TEST_P(DataBaseTest, light_data_sigle_value) {
    db->SetValue("CartesianGeometry", "hello world!");
    db->SetValue("b", 5.0);
    db->Flush();
    EXPECT_EQ(db->GetValue<std::string>("CartesianGeometry"), "hello world!");
    EXPECT_DOUBLE_EQ(db->GetValue<double>("b"), 5);
    EXPECT_EQ(db->size(), 2);
    std::cout << m_url << " :  " << (*db) << " " << std::endl;
}

TEST_P(DataBaseTest, light_data_SetValue_ntuple) {
    db->SetValue("tuple3", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    //    (*db)["strlist"] = {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}};
    db->SetValue("tuple1", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    db->SetValue("Box", {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    db->SetValue("str_tuple", {"wa wa", "la la"});

    db->SetValue("A", {1, 2, 3});
    db->SetValue("C", {{1.0, 2.0, 3.0}, {2.0}, {7.0, 9.0}});

    db->Flush();
    std::cout << m_url << " : " << (*db) << std::endl;

    EXPECT_EQ((db->GetValue<nTuple<Real, 6>>("tuple1")), (nTuple<Real, 6>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ((db->GetValue<nTuple<Real, 2, 3>>("Box")), (nTuple<Real, 2, 3>{{1, 2, 3}, {4, 5, 6}}));
}

TEST_P(DataBaseTest, light_data_multilevel) {
    db->SetValue("a/b/sub/1/2/3/4/d", 5.0);
    db->SetValue("/1/2/3/4/d", 5);
    db->Flush();
    EXPECT_DOUBLE_EQ((db->GetValue<Real>("a/b/sub/1/2/3/4/d")), 5);
    EXPECT_EQ((db->GetValue<int>("/1/2/3/4/d")), 5);

    std::cout << m_url << " : " << (*db) << std::endl;
}
TEST_P(DataBaseTest, light_data_keyvalue) {
    db->SetValue("i", {"default"_, "abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
    db->SetValue("a",
                 {"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
                  "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
    //    (*db)["h"] = {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}};
    db->SetValue("nest", {"abc"_ = {"abc1"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
    EXPECT_TRUE(db->Check("a/a"));
    EXPECT_FALSE(db->Check("a/not_debug"));
    std::cout << m_url << " : " << (*db) << std::endl;
}

TEST_P(DataBaseTest, light_data_AddValue) {
    db->AddValue("a", {0, 5, 3, 4});
    db->AddValue("a", {1, 5, 3, 4});

    db->Flush();
    std::cout << m_url << " : " << (*db) << std::endl;

    EXPECT_EQ((db->GetValue<nTuple<int, 4>>("a/1")), (nTuple<int, 4>{1, 5, 3, 4}));
    EXPECT_EQ((db->GetValue<nTuple<int, 2, 4>>("a")), (nTuple<int, 2, 4>{{0, 5, 3, 4}, {1, 5, 3, 4}}));
}
// TEST_P(DataBaseTest, block_data) {
//    auto db = DataNode::New(m_url);
//}
INSTANTIATE_TEST_CASE_P(DataBaseTestP, DataBaseTest,
                        testing::Values(               //
                            "mem://",                  //
                            "h5://?rw,a=234,b=6#123",  //
                            "imas://",                 //
                            "lua://"));
//
// TEST(DataTable, samrai) {
//    logger::set_stdout_level(1000);
//
//    LOGGER << "Registered DataBase: " << GLOBAL_DATA_BACKEND_FACTORY.GetBackendList() << std::endl;
//    DataTable db("samrai://");
//    //   db->SetEntity("f", {1, 2, 3, 4, 5, 56, 6, 6});
//    //   db->SetEntity("/d/e/f", "Just atest");
//    //   db->SetEntity("/d/e/g", {"a"_ = "la la land", "b"_ = 1235.5});
//    //   db->SetEntity("/d/e/e", 1.23456);
//   db->SetEntity("box", {{1, 2, 3}, {4, 5, 6}});
//    LOGGER << *db.database() << std::endl;
//    LOGGER << "box  = " <<db->GetEntity<std::tuple<nTuple<int, 3>, nTuple<int, 3>>>("box") << std::endl;
//
//}
