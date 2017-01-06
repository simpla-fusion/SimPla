//
// Created by salmon on 17-1-6.
//

#include <simpla/SIMPLA_config.h>
#include <simpla/data/all.h>

#include <complex>
#include <iostream>


using namespace simpla;
using namespace simpla::data;
int main(int argc, char** argv) {

    DataTable db;

    db.set_value("CartesianGeometry.name", "hello world!");

    std::cout<<db<<std::endl;
}