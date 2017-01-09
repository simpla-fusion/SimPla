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

    db.set_value("CartesianGeometry.name", "hello world!");
    db.set_value("CartesianGeometry.value", 1.0234);
    db.set_value("CartesianGeometry.t.second", 2);
    db.set_value("CartesianGeometry.vec3", nTuple<Real,3>{2,3,2});
    std::cout << db << std::endl;
}