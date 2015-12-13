/**
 * @file mesh_layout_test.cpp
 * @author salmon
 * @date 2015-12-13.
 */

#include <gtest/gtest.h>
#include "../mesh/mesh_layout.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/ntuple_ext.h"
#include "../../gtl/utilities/log.h"
#include "../../gtl/iterator/range.h"


#include "../../io/io.h"

#include "../../field/field.h"

#include "../../manifold/manifold_traits.h"

#include "../../gtl/utilities/log.h"

#include "../pre_define/predefine.h"

using namespace simpla;

TEST(MeshMultiBlock, RectMesh)
{
    typedef mesh::MeshPatch<manifold::CartesianManifold> mesh_type;

    mesh_type m;

    traits::field_t<Real, mesh_type, VERTEX> f0{m};
    traits::field_t<Real, mesh_type, EDGE> f1a{m};
    traits::field_t<Real, mesh_type, EDGE> f1b{m};

    f0.clear();
    f1a.clear();
    f1b.clear();


    m.time_integral(1.0, f1a, f1b * f0);


}