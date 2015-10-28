/**
 * @file geometry_test.cpp
 * @author salmon
 * @date 2015-10-28.
 */


#include <gtest/gtest.h>

#include "../../gtl/ntuple.h"
#include "../../gtl/ntuple_ext.h"
#include "../../gtl/utilities/log.h"
#include "../../gtl/iterator/range.h"

#include "../geometry/geometry.h"
#include "../../geometry/cs_cartesian.h"
#include "../topology/rectmesh.h"

#include "../../io/io.h"

#include "../../field/field_dense.h"
#include "../../field/field_expression.h"
#include "../../field/field_traits.h"

#include "../../manifold/domain.h"
#include "../../manifold/manifold_traits.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"

#include "../../gtl/utilities/log.h"
#include "../../manifold/pre_define/cartesian.h"

using namespace simpla;

TEST(GeometryTest, RectMesh)
{
    typedef Real value_type;

    manifold::Cartesian<3, topology::RectMesh> mesh;

    nTuple<size_t, 3> dim = {5, 5, 5};

    nTuple<size_t, 3> xmin = {0, 0, 0};
    nTuple<size_t, 3> xmax = {1, 2, 3};

    mesh.dimensions(dim);

    mesh.box(xmin, xmax);

    mesh.deploy();

    auto f0 = traits::make_field<VERTEX, value_type>(mesh);
    auto f1 = traits::make_field<EDGE, value_type>(mesh);
    auto f1b = traits::make_field<EDGE, value_type>(mesh);
    f0.clear();
    f1.clear();
    f1 = grad(f0);
}