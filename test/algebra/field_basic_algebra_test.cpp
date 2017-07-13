/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include "field_basic_algebra_test.h"
#include <gtest/gtest.h"
#include "simpla/predefine/mesh/CartesianGeometry.h"
typedef simpla::mesh::CartesianGeometry mesh_type;

using namespace simpla;

typedef testing::Types<                //
    Field<mesh_type, Real, VERTEX>,    //
    Field<mesh_type, Real, EDGE>,      //
    Field<mesh_type, Real, FACE>,      //
    Field<mesh_type, Real, VOLUME>,    //
    Field<mesh_type, Real, VERTEX, 3>  //   ,
                                       //    Field<mesh_type, Real, EDGE, 3>,    //
                                       //    Field<mesh_type, Real, FACE, 3>,    //
                                       //    Field<mesh_type, Real, VOLUME, 3>   //
    >
    TypeParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);
