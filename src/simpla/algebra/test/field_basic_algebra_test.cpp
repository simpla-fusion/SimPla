/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <simpla/mesh/DummyMesh.h>
#include <simpla/algebra/Field.h>
#include "field_basic_algebra_test.h"

using namespace simpla;

typedef mesh::DummyMesh mesh_type;


typedef testing::Types< //
        Field<Real, mesh_type, VERTEX>//
        , Field<Real, mesh_type, EDGE>//
        , Field<Real, mesh_type, FACE>//
        , Field<Real, mesh_type, VOLUME>//

        , Field<Real, mesh_type, VERTEX, 3>//
        , Field<Real, mesh_type, EDGE, 3> //
        , Field<Real, mesh_type, FACE, 3> //
        , Field<Real, mesh_type, VOLUME, 3> //

> TypeParamList;


INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

