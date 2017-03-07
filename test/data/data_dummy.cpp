//
// Created by salmon on 17-1-6.
//

#include <simpla/SIMPLA_config.h>

#include <simpla/algebra/all.h>
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
    logger::set_stdout_level(1000);

    //    if (argc > 1) {
    //        DataTable lua_db(argv[1]);
    //        CHECK(lua_db.Get("AAA").as<int>());
    //        CHECK(lua_db.Get("CCC").as<double>());
    //    } else
    {
        DataTable db;
        db.Put("CartesianGeometry.name2"_ = std::string("hello world!"));
        db.Put("a"_, "not_debug"_ = false,
               "c"_ =
                   {
                       1_ = "hello world!",  //
                       2_ = 1.0234           //,
                                             //               "t.second"_ = 2,                    //
                                             //               "vec3"_ = nTuple<Real, 3>{2, 3, 2}  //
                   }                         //            , "is_test3"_ = {1, 3, 4}
               );
        //        CHECK(db.Get("CartesianGeometry.name2").as<std::string>());
        //        db.Put({"Check"});
        //        std::cout << db << std::endl;

        db.Put("b.sub.d", nTuple<int, 3>{1, 2, 3});
        db.Put("b.sub.e", nTuple<int, 4>{1, 2, 3, 4});
        CHECK(db.Get("a").as<bool>(false));
        db.Put("A", 1);
        CHECK(db.Get("A").as<int>(1));
        db.Put("A", 1.1257);
        CHECK(db.Get("A").as<double>());

//        std::cout << db << std::endl;
    }
}