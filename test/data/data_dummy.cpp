//
// Created by salmon on 17-1-6.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
//#include <simpla/algebra/all.h>
//#include <simpla/data/DataEntity.h>
//#include <simpla/toolbox/DataTableLua.h>
#include <simpla/data/DataBackendFactroy.h>
#include <simpla/data/DataEntity.h>
#include <simpla/data/DataEntityArray.h>
#include <simpla/data/DataTable.h>
#include <complex>
#include <iostream>

using namespace simpla;
using namespace simpla::data;

int main(int argc, char** argv) {
    //    DataTable db;
    logger::set_stdout_level(1000);
    //
    //    //    db.set("CartesianGeometry.name2"_ = std::string("hello world!"));
    //    db.Set("a"_, "not_debug"_ = false,
    //           "c"_ =
    //               {
    //                   1_ = "hello world!",  //
    //                   2_ = 1.0234           //,
    //                                         //               "t.second"_ = 2,                    //
    //                                         //               "vec3"_ = nTuple<Real, 3>{2, 3, 2}  //
    //               }                         //            , "is_test3"_ = {1, 3, 4}
    //           );
    //    //    db.Set({"Check"});
    //    db.SetValue("b.sub.c", 1);
    //    db.SetValue("b.sub.d", nTuple<int, 3>{1, 2, 3});
    //    db.SetValue("b.sub.e", nTuple<int, 4>{1, 2, 3, 4});
    //    std::cout << db << std::endl;
    //    std::cout << db.GetValue<bool>("a") << std::endl;
    //    CHECK(db.GetValue<int>("b.sub.c"));
    //    db.SetValue("b.sub.c", 1.1257);
    //    CHECK(db.GetValue<double>("b.sub.c"));
    //    //    CHECK(db.GetValue<int>("b.sub.c"));

    if (argc > 1) {
        DataTable lua_db(argv[1]);
        //        CHECK(lua_db.GetValue<int>("AAA"));
        //        CHECK(lua_db.GetValue<double>("epsilon0"));
        //        CHECK(lua_db.GetValue<int>("CCC.a.b"));
        CHECK(lua_db["CCC.c"].GetValue<int>());
    }
}