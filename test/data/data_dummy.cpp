//
// Created by salmon on 17-1-6.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <complex>
#include <iostream>

using namespace simpla;
using namespace simpla::data;

int main(int argc, char** argv) {
    logger::set_stdout_level(1000);

    //    if (argc > 1) {
    //        DataTable lua_db(argv[1]);
    //        CHECK(lua_db.Get("AAA").as<int>());
    //        CHECK(lua_db.Get("CCC").as<double>());
    //    } else
    {
        DataTable db(argv[1]);
//        LOGGER << "AAA =" << (*db.Get("AAA")) << std::endl;
//        LOGGER << "CCC =" << (*db.Get("CCC")) << std::endl;
//        db.Set("CartesianGeometry", "hello world!");
        LOGGER << "AAA =" << db << std::endl;

        //        db.Set("d", {1, 2, 3, 4, 5, 56, 6, 6});
        //        db.Set("g", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
        //        db.Set("e", {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}});
        //        db.Set("f", {"a"_, "not_debug"_ = false,
        //                     "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello
        //                     !"}});
        //        db.Set("h", {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}});
        //        db.Set("i", {"abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
        //        db.Set("j", {"abc"_ = {"abc"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
        //
        ////        LOGGER << (db) << std::endl;
        //
        //        db.Set("b.sub.d", nTuple<int, 3>{1, 2, 3});
        //        db.Set("b.sub.e", nTuple<int, 4>{1, 2, 3, 4});
        //        LOGGER << "a =" << (db.Get("a")->as<bool>(false)) << std::endl;
        //
        //        LOGGER << "b.sub.e  = " << ((db.Get("b.sub.e")->as<nTuple<int, 4>>())) << std::endl;
        //
        //        db.Set("A", 3);
        //        LOGGER << "A = " << (db.Get("A")->as<int>()) << std::endl;
        //
        //        db.Set("A", 1.1257);
        //        LOGGER << "A = " << (db.Get("A")->as<double>()) << std::endl;

        //        std::cout << db << std::endl;
    }
}