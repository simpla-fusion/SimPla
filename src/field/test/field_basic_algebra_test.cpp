/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <gtest/gtest.h>


#include "../../field/Field.h"
//#include "../../manifold/pre_define/PreDefine.h"

#include "field_basic_algebra_test.h"

using namespace simpla;

typedef mesh::CoRectMesh mesh_type;

//typedef manifold::CartesianManifold mesh_type;

typedef testing::Types< //

        FieldAttr<double, mesh_type, mesh::VERTEX>,//
        FieldAttr<double, mesh_type, mesh::EDGE>, //
        FieldAttr<double, mesh_type, mesh::FACE>, //
        FieldAttr<double, mesh_type, mesh::VOLUME>, //

        FieldAttr<Vec3, mesh_type, mesh::VERTEX>, //
        FieldAttr<Vec3, mesh_type, mesh::EDGE>, //
        FieldAttr<Vec3, mesh_type, mesh::FACE>, //
        FieldAttr<Vec3, mesh_type, mesh::VOLUME>  //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::mesh_type> //
        TestField<TF>::mesh = std::make_shared<typename TestField<TF>::mesh_type>();

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

