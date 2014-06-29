/*
 * iterpolator_test.cpp
 *
 *  Created on: 2014年6月29日
 *      Author: salmon
 */

#include "iterpolator_test.h"

#include "mesh_rectangle.h"
#include "geometry_cartesian.h"
#include "geometry_cylindrical.h"
#include "uniform_array.h"

typedef ::testing::Types<

Mesh<CartesianGeometry<UniformArray>, false>     //
,Mesh<CartesianGeometry<UniformArray>, true>    //
,Mesh<CylindricalGeometry<UniformArray>, true>   //
,Mesh<CylindricalGeometry<UniformArray>, false>  //

> MeshTypeList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestIterpolator, MeshTypeList);
