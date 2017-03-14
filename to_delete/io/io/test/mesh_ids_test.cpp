/**
 * @file mesh_ids_test.cpp.cpp
 * @author salmon
 * @date 2015-10-26.
 */

#include <gtest/gtest.h>

#include "simpla/algebra/nTuple.h"
#include "simpla/algebra/nTupleExt.h"
#include "../../toolbox/utilities/Log.h"
#include "../mesh/mesh_ids.h"
#include "Parallel.h"

using namespace simpla;

TEST(GeometryTest, MeshIDs)
{

    typedef mesh::MeshIDs_<4> m;

    typedef typename mesh::MeshIDs_<4>::id_type id_type;


    nTuple<long, 4> b = {0, 0, 0};
    nTuple<long, 4> e = {3, 4, 2};

    mesh::MeshIDs_<4>::range_type r(b, e, FACE);

//    auto ib = r.begin();
//
//    auto ie = r.end();
//    for (int i = 0; i < 40; ++i)
//    {
//        CHECK(ib.m_self_);
//        ++ib;
//    }
    parallel::parallel_for(r, [&](mesh::MeshIDs_<4>::range_type const &r0)
    {

        CHECK(r0.size());
    });


    size_t res = parallel::parallel_reduce(r, 0UL,
                                           [&](mesh::MeshIDs_<4>::range_type const &r, size_t init) -> size_t
                                           {
                                               for (auto const &s:r)
                                               {
                                                   ++init;
                                               }

                                               return init;
                                           },
                                           [](size_t x, size_t y) -> size_t
                                           {
                                               return x + y;
                                           }
    );

    CHECK(res);

}
//
//TEST(GeometryTest, structured_mesh)
//{
//    mesh_as::CartesianGeometry t;
//
//    nTuple<size_t, 3> b = {3, 4, 1};
//
//    t.dimensions(b);
//
//    t.save_mesh();
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
//    nTuple<size_t, 3> topology_dims = {5, 5, 5};
//
//    t.dimensions(topology_dims);
//
//    t.save_mesh();
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
//    Geometry < Metric < coordinate_system::Cylindrical < 2 >> , topology_dims::RectMesh<>>
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
//    g.save_mesh();
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
//    BaseManifold<coordinate_system::Cartesian<3, 2>, mesh_as::tags::CartesianGeometry> g;
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
//    g.save_mesh();
//
//    CHECK(g.dx());
//
//    for (auto const &s:g.range<VERTEX>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.point_global_to_local(g.point(s)) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<EDGE>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.point_global_to_local(g.point(s)) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<FACE>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.point_global_to_local(g.point(s)) << std::endl;
//    }
//    CHECK("=======================");
//    for (auto const &s:g.range<VOLUME>())
//    {
//        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack_index(s) << " [" <<
//        g.sub_index(s) << "] p=" << g.point(s) << " (s,r)=" << g.point_global_to_local(g.point(s)) << std::endl;
//    }
//}