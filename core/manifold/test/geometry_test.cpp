/**
 * @file geometry_test.cpp
 * @author salmon
 * @date 2015-10-28.
 */


#include <gtest/gtest.h>

#include "../../gtl/ntuple.h"
#include "../../gtl/ntuple_ext.h"
#include "../../gtl/utilities/Log.h"
#include "../../gtl/iterator/range.h"


#include "../../io/IO.h"

#include "../../field/field.h"

#include "manifold_traits.h"

#include "../../gtl/utilities/Log.h"

#include "PreDefine.h"

using namespace simpla;

TEST(GeometryTest, RectMesh)
{
    LOGGER.set_stdout_visable_level(10);


    typedef Real value_type;

    manifold::Cartesian<3, topology::RectMesh<>> mesh;

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

    LOG_CMD(f1 = grad(f0));
}