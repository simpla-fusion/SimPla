/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <simpla/algebra/Field.h>
#include <simpla/manifold/pre_define/PreDefine.h>
#include "field_basic_algebra_test.h"

using namespace simpla;

typedef manifold::CartesianManifold mesh_type;


typedef testing::Types< //
       field_t < Real, mesh_type, mesh::VERTEX>//
, field_t <Real, mesh_type, mesh::EDGE>//
, field_t <Real, mesh_type, mesh::FACE>//
, field_t <Real, mesh_type, mesh::VOLUME>//

, field_t<Real, mesh_type, mesh::VERTEX, 3>//
, field_t<Real, mesh_type, mesh::EDGE, 3> //
, field_t<Real, mesh_type, mesh::FACE, 3> //
, field_t<Real, mesh_type, mesh::VOLUME, 3> //

>
TypeParamList;


INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

