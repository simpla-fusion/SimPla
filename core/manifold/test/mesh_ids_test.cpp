/**
 * @file mesh_ids_test.cpp.cpp
 * @author salmon
 * @date 2015-10-26.
 */


#include "../../gtl/ntuple.h"
#include "../../gtl/ntuple_ext.h"
#include "../../gtl/utilities/log.h"
#include "../topology/mesh_ids.h"

using namespace simpla;

int main(int argc, char **argv)
{
    std::cout << "Hello world" << std::endl;


    typedef MeshIDs_<4> m;

    nTuple<long, 4> b = {0, 0, 0};
    nTuple<long, 4> e = {3, 4, 5};

    MeshIDs_<4>::iterator it(b, b, e, FACE);
    MeshIDs_<4>::iterator ib(it);
    std::cout << "Hello world" << std::endl;

    for (int i = 0; i < 200; ++i)
    {
        ++it;
        std::cout << "[" << it - ib << "]" << m::unpack_index(*it) << m::sub_index(*it) << std::endl;
    }

    it = ib + 16;
    std::cout << it - ib << std::endl;
//    for (int i = 0; i < 200; ++i)
//    {
//        --it;
//        std::cout << m::unpack_index(*it) << m::sub_index(*it) << std::endl;
//    }
//
//    for (int i = 0; i < 10; ++i)
//    {
//        it += i;
//        std::cout << m::unpack_index(*it) << m::sub_index(*it) << std::endl;
//    }
//    for (int i = 0; i < 10; ++i)
//    {
//        it -= i;
//        std::cout << m::unpack_index(*it) << m::sub_index(*it) << std::endl;
//    }
}