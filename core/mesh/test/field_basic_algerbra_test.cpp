/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>

#include "field_basic_algerbra_test.h"
#include "../structured/structured.h"

using namespace simpla;
typedef CartesianRectMesh mesh_type;
typedef testing::Types< //

		typename mesh_type::template field<VERTEX, double>, //
		typename mesh_type::template field<EDGE, double>, //
		typename mesh_type::template field<FACE, double>, //
		typename mesh_type::template field<VOLUME, double>, //

		typename mesh_type::template field<VERTEX, Vec3>, //
		typename mesh_type::template field<EDGE, Vec3>, //
		typename mesh_type::template field<FACE, Vec3>, //
		typename mesh_type::template field<VOLUME, Vec3>  //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::mesh_type> //
TestField<TF>::mesh = std::make_shared<typename TestField<TF>::mesh_type>( );

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

