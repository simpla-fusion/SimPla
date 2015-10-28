/**
 * @file mesh_ids_test.cpp.cpp
 * @author salmon
 * @date 2015-10-26.
 */

#include <gtest/gtest.h>

#include "../../gtl/ntuple.h"
#include "../../gtl/ntuple_ext.h"
#include "../../gtl/utilities/log.h"
#include "../../gtl/iterator/range.h"
#include "../topology/mesh_ids.h"
#include "co_rect_mesh.h"
#include "../geometry/geometry.h"
#include "../../geometry/cs_cartesian.h"

using namespace simpla;

TEST(GeometryTest, MeshIDs)
{
    std::cout << "Hello world" << std::endl;


    typedef MeshIDs_<4> m;

    nTuple<long, 4> b = {0, 0, 0};
    nTuple<long, 4> e = {3, 4, 5};

    MeshIDs_<4>::iterator it(b, b, e, FACE);
    auto ib = MeshIDs_<4>::iterator(b, b, e, VERTEX);
    auto ie = MeshIDs_<4>::iterator(e - 1, b, e, VERTEX) + 1;
    std::cout << "Hello world" << std::endl;
    CHECK((ib.template block_iterator<long, 4>::operator*()));
    CHECK((ie.template block_iterator<long, 4>::operator*()));
//    for (int i = 0; i < 200; ++i)
//    {
//        ++it;
//        std::cout << "[" << it - ib << "]" << m::unpack_index(*it) << m::sub_index(*it) << std::endl;
//    }

    it = ib + 16;
    std::cout << it - ib << std::endl;

    auto r = MeshIDs_<4>::template range<VERTEX>(b, e);

    CHECK(r.size());

    size_t count = 0;
    CHECK((r.begin().template block_iterator<long, 4>::operator*()));
    CHECK((r.end().template block_iterator<long, 4>::operator*()));

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

TEST(GeometryTest, structured_mesh)
{
    RectMesh t;

    nTuple<size_t, 3> b = {3, 4, 1};

    t.dimensions(b);

    t.deploy();

    for (auto const &s:t.range<VERTEX>())
    {
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
    CHECK("=======================");

    for (auto const &s:t.range<EDGE>())
    {
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
    CHECK("=======================");


    for (auto const &s:t.range<FACE>())
    {
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
    CHECK("=======================");

    for (auto const &s:t.range<VOLUME>())
    {
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
}


TEST(GeometryTest, Coordinates)
{
    RectMesh t;

    nTuple<size_t, 3> dims = {5, 5, 5};

    t.dimensions(dims);

    t.deploy();

    for (auto const &s0:t.range<VERTEX>())
    {
        auto s = s0 - RectMesh::_DI;
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
    CHECK("=======================");

    for (auto const &s0:t.range<EDGE>())
    {
        auto s = s0 - RectMesh::_DI;
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
    CHECK("=======================");


    for (auto const &s0:t.range<FACE>())
    {
        auto s = s0 - RectMesh::_DI;
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
    CHECK("=======================");

    for (auto const &s0:t.range<VOLUME>())
    {
        auto s = s0 - RectMesh::_DI;
        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
        t.sub_index(s) << "]" << std::endl;
    }
}


TEST(GeometryTest, Geometry)
{
    Geometry<coordinate_system::Cartesian<3, 2>, topology::tags::CoRectMesh> g;

    nTuple<size_t, 3> dim = {10, 1, 1};

    nTuple<size_t, 3> xmin = {0, 0, 0};
    nTuple<size_t, 3> xmax = {1, 1, 1};

    g.dimensions(dim);

    g.box(xmin, xmax);

    g.deploy();

    CHECK(g.dx());

    for (auto const &s:g.range<VERTEX>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] v=" << g.volume(s) << std::endl;
    }
    CHECK("=======================");
    for (auto const &s:g.range<EDGE>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] v=" << g.volume(s) << std::endl;
    }
    CHECK("=======================");
    for (auto const &s:g.range<FACE>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] v=" << g.volume(s) << std::endl;
    }
    CHECK("=======================");
    for (auto const &s:g.range<VOLUME>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] v=" << g.volume(s) << std::endl;
    }
}


TEST(GeometryTest, CoordinateSystem)
{
    Geometry<coordinate_system::Cartesian<3, 2>, topology::tags::CoRectMesh> g;

    nTuple<size_t, 3> dim = {5, 5, 5};

    nTuple<size_t, 3> xmin = {0, 0, 0};
    nTuple<size_t, 3> xmax = {1, 2, 3};

    g.dimensions(dim);

    g.box(xmin, xmax);

    g.deploy();

    CHECK(g.dx());

    for (auto const &s:g.range<VERTEX>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
    }
    CHECK("=======================");
    for (auto const &s:g.range<EDGE>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
    }
    CHECK("=======================");
    for (auto const &s:g.range<FACE>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
    }
    CHECK("=======================");
    for (auto const &s:g.range<VOLUME>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
    }
}