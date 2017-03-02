//
// Created by salmon on 17-1-6.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/toolbox/DataTableLua.h>

#include <complex>
#include <iostream>
using namespace simpla;
using namespace simpla::data;

int main(int argc, char** argv) {
    DataTable db;
    logger::set_stdout_level(1000);

    //    db.set("CartesianGeometry.name2"_ = std::string("hello world!"));
    db.SetValue({"is_test", "not_debug"_ = false,
                 "CartesianGeometry"_ = {
                     "GetName"_ = "hello world!",  //
                     "is_test",
                     "value"_ = 1.0234,                  //
                     "t.second"_ = 2,                    //
                     "vec3"_ = nTuple<Real, 3>{2, 3, 2}  //
                 }});
    db.SetValue({"Check"});
    std::cout << db << std::endl;

    if (argc > 1) {
        DataTableLua lua_db;
        lua_db.ParseFile(argv[1]);
        std::cout << "AAA=" << lua_db.GetValue<int>("AAA") << std::endl;
    }
}