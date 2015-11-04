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

#include "../../gtl/utilities/log.h"

#include "../geometry/geometry.h"
#include "../../geometry/cs_cartesian.h"
#include "../../geometry/cs_cylindrical.h"
#include "../topology/rectmesh.h"
#include "../topology/mesh_ids.h"
#include "../topology/corect_mesh.h"

using namespace simpla;
//
//TEST(GeometryTest, MeshIDs)
//{
//
//    typedef MeshIDs_<4> m;
//
//    typedef Metric<coordinate_system::template Cartesian<3> > metric_type;
//
//    nTuple<long, 3> m_min_ = {0, 0, 0};
//
//    nTuple<Real, 3> p[m::NUM_OF_NODE_ID];
//
//    id_type ss[m::NUM_OF_NODE_ID];
//
//    m::get_adjoin_vertices(m::TAG_VOLUME, m::pack_index(m_min_), ss);
//
//    for (int i = 0; i < m::NUM_OF_NODE_ID; ++i)
//    {
//        p[i] = m::point(ss[i]);
////        CHECK(m::unpack(ss[i]));
//        std::cout << "[" << i << "]" << p[i] << std::endl;
//    }
//
//
//    CHECK(metric_type::metric_length(p[0], p[1]));
//
//    CHECK(metric_type::metric_length(p[0], p[3]));
//
//    CHECK(metric_type::metric_length(p[0], p[4]));
//
//    CHECK(metric_type::metric_area(p[0], p[1], p[2]));
//
//    CHECK(metric_type::metric_area(p[0], p[1], p[2]));
//
//
////
////    nTuple<Real, 3> p[m::NUM_OF_NODE_ID];
////    id_type ss[m::NUM_OF_NODE_ID];
////
////    m::get_adjoin_vertices(m::TAG_VOLUME, m::pack_index(m_min_), ss);
////
////    CHECK_BIT(m::ID_MASK);
////    CHECK_BIT(m::OVERFLOW_FLAG);
////    CHECK_BIT(LI);
////    CHECK_BIT(LI & m::ID_MASK);
////    CHECK_BIT((LI) |
////              (((LI & (1UL << (m::ID_DIGITS - 2))) == 0) ? 0UL : (static_cast<id_type>( -1L << (m::ID_DIGITS - 1)))));
////    CHECK_BIT(m::extent_flag_bit(LI));
////    CHECK_BIT(m::unpack_id(LI, 0));
////
////    CHECK(m::unpack(s + LI));
////    CHECK(m::unpack(s + HI));
////
////    CHECK(m::unpack_index(s + LI));
////    CHECK(m::unpack_index(s + HI));
////
////    CHECK(m::point(s + LI));
////    CHECK(m::point(s + HI));
//
//
//
//
////    for (int i = 0; i < m::NUM_OF_NODE_ID; ++i)
////    {
////        auto I = m::unpack(ss[i]);
////
////        CHECK_BIT(I[0]);
////        CHECK_BIT((I[0] + (m::ID_MASK >> 1)) & (~m::OVERFLOW_FLAG));
////        CHECK_BIT((I[0] + (m::ID_MASK >> 1)) & (~m::OVERFLOW_FLAG) - (m::OVERFLOW_FLAG << 1));
////
////        CHECK(static_cast<Real>( I[0] + (m::ID_MASK >> 1)) - (m::OVERFLOW_FLAG << 1));
////
////    }
//    nTuple<long, 4> b = {0, 0, 0};
//    nTuple<long, 4> e = {3, 4, 5};
//
//    MeshIDs_<4>::iterator it(b, b, e, FACE);
//    auto ib = MeshIDs_<4>::iterator(b, b, e, VERTEX);
//    auto ie = MeshIDs_<4>::iterator(e - 1, b, e, VERTEX) + 1;
//    std::cout << "Hello world" << std::endl;
//    CHECK((ib.template block_iterator<long, 4>::operator*()));
//    CHECK((ie.template block_iterator<long, 4>::operator*()));
////    for (int i = 0; i < 200; ++i)
////    {
////        ++it;
////        std::cout << "[" << it - ib << "]" << m::unpack_index(*it) << m::sub_index(*it) << std::endl;
////    }
//
//    it = ib + 16;
//    std::cout << it - ib << std::endl;
//
//    auto r = MeshIDs_<4>::template make_range<VERTEX>(b, e);
//
//    CHECK(r.size());
//
//    size_t count = 0;
//    CHECK((r.begin().template block_iterator<long, 4>::operator*()));
//    CHECK((r.end().template block_iterator<long, 4>::operator*()));

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
//}
//
//TEST(GeometryTest, structured_mesh)
//{
//    topology::CoRectMesh t;
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
TEST(GeometryTest, Geometry)
{
    Geometry<Metric<coordinate_system::Cylindrical<2>>, topology::RectMesh<>> g;

    nTuple<size_t, 3> dim = {1, 100, 1};
    nTuple<Real, 3> xmin = {1, 0, 0};
    nTuple<Real, 3> xmax = {2, 1, TWOPI};

    g.dimensions(dim);

    g.box(xmin, xmax);

    g.deploy();

    auto dx = g.dx();


    for (auto const &s:g.range<VERTEX>())
    {
        std::cout << "[" << g.iform(s) << "\t," << g.hash(s) << "] \t " << g.unpack(s) <<
        " " << g.unpack_index(s) << " " << g.point(s) << " [" << g.sub_index(s) << "] v=" << g.volume(s) << " \t" <<
        g.inv_volume(s) << " \t" <<
        g.dual_volume(s) << " \t" <<
        g.inv_dual_volume(s) << std::endl;


    }
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
}
//
//
//TEST(GeometryTest, CoordinateSystem)
//{
//    Geometry<coordinate_system::Cartesian<3, 2>, topology::tags::CoRectMesh> g;
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