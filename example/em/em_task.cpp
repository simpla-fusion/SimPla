/**
 * @file multi_task.cpp
 * @author salmon
 * @date 2015-11-21.
 */

//#include <tbb/task.h>
//#include <tbb/task_group.h>
#include <iostream>
#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/primitives.h"
#include "../../core/gtl/utilities/pretty_stream.h"
#include "../../core/dataset/datatype.h"

using namespace simpla;

int main(int argc, char **argv)
{
//    tbb::task_group group;
//
//    group.run([&]() { std::cout << "First" << std::endl; });
//    group.run([&]() { std::cout << "Second" << std::endl; });
//    group.run([&]() { std::cout << "Third" << std::endl; });
//
//    group.wait();


    auto dtype = traits::datatype<nTuple<Real, 3 >>::create();

    std::cout << dtype.name() << std::endl;
    std::cout << dtype.rank() << std::endl;
    std::cout << dtype.extents() << std::endl;
}

