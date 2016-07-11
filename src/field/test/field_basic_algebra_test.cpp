/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "../../field/Field.h"
#include "../../manifold/pre_define/PreDefine.h"
#include "field_basic_algebra_test.h"

using namespace simpla;

typedef manifold::CartesianManifold mesh_type;


typedef testing::Types< //

        field_t<Real, mesh_type, mesh::VERTEX>//
        , field_t<Real, mesh_type, mesh::EDGE>//
        , field_t<Real, mesh_type, mesh::FACE>//
        , field_t<Real, mesh_type, mesh::VOLUME>//

//        , field_t<Vec3, mesh_type, mesh::VERTEX>//
//        , field_t<Vec3, mesh_type, mesh::EDGE> //
//        , field_t<Vec3, mesh_type, mesh::FACE> //
//        , field_t<Vec3, mesh_type, mesh::VOLUME>  //

> TypeParamList;


INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

