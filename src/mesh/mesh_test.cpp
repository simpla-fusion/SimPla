/*
 * mesh_test.cpp
 *
 *  Created on: 2014年6月27日
 *      Author: salmon
 */

#include "mesh_test.h"

#include "mesh_rectangle.h"
#include "geometry_cartesian.h"
#include "geometry_cylindrical.h"
#include "uniform_array.h"

typedef ::testing::Types<

Mesh<CartesianGeometry<UniformArray>, true>,

Mesh<CartesianGeometry<UniformArray>, false>,

Mesh<CylindricalGeometry<UniformArray, 2>, true>,

Mesh<CylindricalGeometry<UniformArray, 2>, false>

> MeshTypeList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestMesh, MeshTypeList);
