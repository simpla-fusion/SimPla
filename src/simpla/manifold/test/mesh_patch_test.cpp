/**
 * @file mesh_layout_test.cpp
 * @author salmon
 * @date 2015-12-13.
 */

#include <gtest/gtest.h>
#include "Mesh"
#include "PreDefine.h"

#include "../../field/field.h"
#include "../../toolbox/utilities/Log.h"
#include "../../toolbox/XDMFStream.h"

using namespace simpla;

TEST(MeshMultiBlock, RectMesh)
{
    typedef manifold::CartesianManifold mesh_type;

    io::XDMFStream out_stream;
    out_stream.open(("test_amr"), "GEqdsk", 0);

    mesh_type m;
    m.box(std::make_tuple(nTuple<Real, 3> {0, 0, 0}, nTuple<Real, 3> {1, 1, 1}));
    m.dimensions(nTuple<size_t, 3>{10, 10, 10});
    m.deploy();

    auto l_box = m.box();

    auto it = m.new_patch(l_box);


    out_stream.open_grid("MultiBlock", 0, io::XDMFStream::TREE);

    out_stream.set_grid(m);

    out_stream.close_grid();

    out_stream.close();


//    traits::field_t<Real, mesh_type, VERTEX> f0{m};
//    traits::field_t<Real, mesh_type, EDGE> f1a{m};
//    traits::field_t<Real, mesh_type, EDGE> f1b{m};
//
//    f0.clear();
//    f1a.clear();
//    f1b.clear();
//    default_time_integral(
//            m,
//            [&](Real dt, traits::field_t<Real, mesh_type, VERTEX> F0,
//                traits::field_t<Real, mesh_type, EDGE> F1a,
//                traits::field_t<Real, mesh_type, EDGE> F1b) { F1b += F1a * F0 * dt; },
//            m.dt(),
//            f0, f1a, f1b
//
//    );

//    m.erase_patch(std::get<0>(it));


}