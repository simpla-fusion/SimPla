/**
 * @file mesh_ids_test.cpp.cpp
 * @author salmon
 * @date 2015-10-26.
 */

#include <gtest/gtest.h>

#include "../../gtl/ntuple.h"
#include "../../gtl/ntuple_ext.h"
#include "../../gtl/utilities/log.h"
#include "../mesh/mesh_ids.h"

using namespace simpla;

TEST(GeometryTest, MeshIDs)
{

    typedef mesh::MeshIDs_<4> m;

    typedef typename mesh::MeshIDs_<4>::id_type id_type;


    nTuple<long, 4> b = {0, 0, 0};
    nTuple<long, 4> e = {3, 4, 2};

    mesh::MeshIDs_<4>::range_type r(b, e, FACE);

    auto ib = r.begin();

    auto ie = r.end();
    for (int i = 0; i < 40; ++i)
    {
        CHECK(ib.m_self_);
        ++ib;
    }

}
//
//TEST(GeometryTest, structured_mesh)
//{
//    mesh::CoRectMesh t;
//
//    nTuple<size_t, 3> b = {3, 4, 1};
//
//    t.dimensions(b);
//
//    t.deploy();
//
//    for (auto const &s:t.range<VERTEX>())
//    {
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//    CHECK("=======================");
//
//    for (auto const &s:t.range<EDGE>())
//    {
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//    CHECK("=======================");
//
//
//    for (auto const &s:t.range<FACE>())
//    {
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//    CHECK("=======================");
//
//    for (auto const &s:t.range<VOLUME>())
//    {
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//}
//
//
//TEST(GeometryTest, Coordinates)
//{
//    RectMesh t;
//
//    nTuple<size_t, 3> dims = {5, 5, 5};
//
//    t.dimensions(dims);
//
//    t.deploy();
//
//    for (auto const &s0:t.range<VERTEX>())
//    {
//        auto s = s0 - RectMesh::_DI;
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//    CHECK("=======================");
//
//    for (auto const &s0:t.range<EDGE>())
//    {
//        auto s = s0 - RectMesh::_DI;
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//    CHECK("=======================");
//
//
//    for (auto const &s0:t.range<FACE>())
//    {
//        auto s = s0 - RectMesh::_DI;
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//    CHECK("=======================");
//
//    for (auto const &s0:t.range<VOLUME>())
//    {
//        auto s = s0 - RectMesh::_DI;
//        std::cout << "[" << t.iform(s) << "\t," << t.hash(s) << "] \t " << t.unpack_index(s) << " [" <<
//        t.sub_index(s) << "]" << std::endl;
//    }
//}
//
//
//TEST(GeometryTest, Geometry)
//{
//    Geometry < Metric < coordinate_system::Cylindrical < 2 >> , topology::RectMesh<>>
//    g;
//
//    nTuple<size_t, 3> dim = {1, 100, 1};
//    nTuple<Real, 3> xmin = {1, 0, 0};
//    nTuple<Real, 3> xmax = {2, 1, TWOPI};
//
//    g.dimensions(dim);
//
//    g.box(xmin, xmax);
//
//    g.deploy();
//
//    auto dx = g.dx();
//
//
//    for (auto const &s:g.range<VERTEX>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack(s) <<
//        " " << g.unpack_index(s) << " " << g.point(s) << " [" << g.sub_index(s) << "] v=" << g.volume(s) << " \t" <<
//        g.inv_volume(s) << " \t" <<
//        g.dual_volume(s) << " \t" <<
//        g.inv_dual_volume(s) << std::endl;
//
//
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<EDGE>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack(s) <<
//        " " << g.unpack_index(s) << " " << g.point(s) << " [" << g.sub_index(s) << "] v=" << g.volume(s) << " \t" <<
//        g.inv_volume(s) << " \t" <<
//        g.dual_volume(s) << " \t" <<
//        g.inv_dual_volume(s) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<FACE>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack(s) <<
//        " " << g.unpack_index(s) << " " << g.point(s) << " [" << g.sub_index(s) << "] v=" << g.volume(s) << " \t" <<
//        g.inv_volume(s) << " \t" <<
//        g.dual_volume(s) << " \t" <<
//        g.inv_dual_volume(s) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<VOLUME>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack(s) <<
//        " " << g.unpack_index(s) << " " << g.point(s) << " [" << g.sub_index(s) << "] v=" << g.volume(s) << " \t" <<
//        g.inv_volume(s) << " \t" <<
//        g.dual_volume(s) << " \t" <<
//        g.inv_dual_volume(s) << std::endl;
//    }
//}
//
//
//TEST(GeometryTest, CoordinateSystem)
//{
//    BaseManifold<coordinate_system::Cartesian<3, 2>, mesh::tags::CoRectMesh> g;
//
//    nTuple<size_t, 3> dim = {5, 5, 5};
//
//    nTuple<size_t, 3> xmin = {0, 0, 0};
//    nTuple<size_t, 3> xmax = {1, 2, 3};
//
//    g.dimensions(dim);
//
//    g.box(xmin, xmax);
//
//    g.deploy();
//
//    CHECK(g.dx());
//
//    for (auto const &s:g.range<VERTEX>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<EDGE>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<FACE>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<VOLUME>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.coordinates_global_to_local(g.point(s)) << std::endl;
//    }
//}