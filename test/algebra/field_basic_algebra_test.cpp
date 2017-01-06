/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include "field_basic_algebra_test.h"
#include <gtest/gtest.h>
#include "simpla/predefine/CartesianGeometry.h"
// typedef simpla::mesh::CartesianGeometry mesh_type;

//#include "simpla/algebra/DummyMesh.h"

typedef simpla::mesh::CartesianGeometry mesh_type;

using namespace simpla;

typedef testing::Types<                 //
    Field<Real, mesh_type, VERTEX>,     //
    Field<Real, mesh_type, EDGE>,       //
    Field<Real, mesh_type, FACE>,       //
    Field<Real, mesh_type, VOLUME>,     //
    Field<Real, mesh_type, VERTEX, 3>,  //
    Field<Real, mesh_type, EDGE, 3>,    //
    Field<Real, mesh_type, FACE, 3>,    //
    Field<Real, mesh_type, VOLUME, 3>   //
    >
    TypeParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);
