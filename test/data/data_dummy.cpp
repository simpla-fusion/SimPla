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
    DataTable db;

    //    db.set("CartesianGeometry.name2"_ = std::string("hello world!"));
    db.insert({"is_test", "not_debug"_ = false,
            "CartesianGeometry"_ = {
                "GetName"_ = "hello world!",  //
                "is_test",
                "value"_ = 1.0234,                  //
                "t.second"_ = 2,                    //
                "vec3"_ = nTuple<Real, 3>{2, 3, 2}  //
            }});
    db.insert({"Check"});
    std::cout << db << std::endl;
}