/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <gtest/gtest.h>


#include "../../field/field.h"
#include "../../geometry/cs_cartesian.h"
#include "../../mesh/default_mesh.h"
#include "field_basic_algebra_test.h"

using namespace simpla;

typedef geometry::coordinate_system::Cartesian<3, 2> cs_type;

typedef DefaultMesh<cs_type> mesh_type;

typedef testing::Types< //

		traits::field_t<mesh_type, VERTEX, double> //,
//		traits::field_t<mesh_type, EDGE, double>, //
//		traits::field_t<mesh_type, FACE, double>, //
//		traits::field_t<mesh_type, VOLUME, double>, //
//
//		traits::field_t<mesh_type, VERTEX, Vec3>, //
//		traits::field_t<mesh_type, EDGE, Vec3>, //
//		traits::field_t<mesh_type, FACE, Vec3>, //
//		traits::field_t<mesh_type, VOLUME, Vec3>  //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::mesh_type> //
		TestField<TF>::mesh = std::make_shared<typename TestField<TF>::mesh_type>();

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

