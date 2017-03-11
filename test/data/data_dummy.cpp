//
// Created by salmon on 17-1-6.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <complex>
#include <iostream>

using namespace simpla;
using namespace simpla::data;

int main(int argc, char** argv) {
    logger::set_stdout_level(1000);

    LOGGER << "Registered DataBackend: " << SingletonHolder<DataBackendFactory>::instance() << std::endl;

    DataTable db;
    if (argc > 1) {
        DataTable lua_db(std::string("lua://") + argv[1]);
        db.Set(lua_db.Get("Context")->cast_as<DataTable>());
        LOGGER << "lua:// " << *lua_db.Get("Context") << std::endl;
    }
    { DataTable samrai_db("samrai://"); }

    db.Set("CartesianGeometry", "hello world!");
    //        LOGGER << "CartesianGeometry: " << *db.Get("CartesianGeometry") << std::endl;

    db.Set("d", {1, 2, 3, 4, 5, 56, 6, 6});
    db.Set("g", {{{1, 2}, {3, 4}}, {{5, 5}, {6, 6}}});
    db.Set("e", {{"abc", "def"}, {"abc", "def"}, {"abc", "def"}, {"abc", "def"}});
    db.Set({"a"_, "not_debug"_ = false, "g"_ = {1, 2, 3, 4, 5, 5, 6, 6},
            "c"_ = {" world!", "hello!", "hello !", "hello!", "hello !", "hello !", "hello !", "hello!"}});
    db.Set("h", {{"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}, {"abc"_ = "def"}});
    db.Set("i", {"abc"_ = 1, "abc"_ = "def", "abc"_ = 2, "abc"_ = "sadfsdf"});
    db.Set("j", {"abc"_ = {"abc"_ = {"def"_ = {"abc"_ = {"abc"_ = "sadfsdf"}}}}});
    db.Set("b.a", 5);
    db.Set("/b/sub/d", nTuple<int, 3>{1, 2, 3});
    db.Set("/b/sub/e", nTuple<int, 4>{1, 2, 3, 4});
    db.Add("/b/sub/c", nTuple<int, 4>{5, 6, 7, 8});
    db.Add("/b/sub/c", nTuple<int, 4>{3, 5, 3, 4});
    db.Add("/b/sub/c", "la la land");
    db.Add("/b/sub/a", {3, 5, 3, 4});
    db.Add("/b/sub/a", 9);

    LOGGER << "b: " << *db.Get("b") << std::endl;
    LOGGER << "db: " << db << std::endl;
    //        LOGGER << "a =" << (db.Get("a")->as<bool>(false)) << std::endl;
    //
    //        LOGGER << "b.sub.e  = " << ((db.Get("b.sub.e")->as<nTuple<int, 4>>())) << std::endl;
    //
    //        db.Set("A", 3);
    //        LOGGER << "A = " << (db.Get("A")->as<int>()) << std::endl;

    LOGGER << "The END !" << std::endl;
}