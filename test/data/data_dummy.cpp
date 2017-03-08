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
        DataTable db;
        db.Set("CartesianGeometry.name2", "hello world!");
        db.Set({"a"_, "not_debug"_ = false,
                "c"_ = {" world!", "hello!", "hello !", "hello !", "hello !", "hello !", "hello !", "hello !"},
                "d"_ = {1, 2, 3, 4, 5, 56, 6, 6},
                "e"_ = {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}}});
        //        CHECK(db.Get("CartesianGeometry.name2").as<std::string>());
        //        db.Put({"Check"});
        LOGGER << (db) << std::endl;

        db.Set("b.sub.d", nTuple<int, 3>{1, 2, 3});
        db.Set("b.sub.e", nTuple<int, 4>{1, 2, 3, 4});
        CHECK(db.Get("a")->as<bool>(false));
        CHECK((db.Get("b.sub.e")->as<nTuple<int, 4>>()));
        db.Set("A", 3);
        CHECK(db.Get("A")->as<int>());
        db.Set("A", 1.1257);
        CHECK(db.Get("A")->as<double>());

        //        std::cout << db << std::endl;
    }
}