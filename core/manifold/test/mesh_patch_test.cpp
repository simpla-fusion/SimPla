/**
 * @file mesh_layout_test.cpp
 * @author salmon
 * @date 2015-12-13.
 */

#include <gtest/gtest.h>
#include "../mesh/mesh_patch.h"
#include "../pre_define/predefine.h"

#include "../../field/field.h"
#include "../../gtl/utilities/log.h"

using namespace simpla;

TEST(MeshMultiBlock, RectMesh)
{
    typedef manifold::CartesianManifold mesh_type;

    mesh_type m;

    auto l_box = m.box();

    traits::field_t<Real, mesh_type, VERTEX> f0{m};
    traits::field_t<Real, mesh_type, EDGE> f1a{m};
    traits::field_t<Real, mesh_type, EDGE> f1b{m};

    f0.clear();
    f1a.clear();
    f1b.clear();

    auto it = m.new_patch(l_box);

    default_time_integral(
            m,
            [&](Real dt, traits::field_t<Real, mesh_type, VERTEX> F0,
                traits::field_t<Real, mesh_type, EDGE> F1a,
                traits::field_t<Real, mesh_type, EDGE> F1b) { F1b += F1a * F0 * dt; },
            m.dt(),
            f0, f1a, f1b

    );

//    m.erase_patch(std::get<0>(it));


}