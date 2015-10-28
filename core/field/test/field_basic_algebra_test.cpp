/**
 * @file field_basic_algebra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <gtest/gtest.h>


#include "../../field/field_comm.h"
#include "../../field/field_dense.h"
#include "../../manifold/domain.h"
#include "../../manifold/pre_define/cartesian.h"

#include "field_basic_algebra_test.h"

using namespace simpla;


typedef manifold::Cartesian<3> mesh_type;

typedef testing::Types< //

        typename traits::field_type<mesh_type, VERTEX, double>::type, //
        typename traits::field_type<mesh_type, EDGE, double>::type, //
        typename traits::field_type<mesh_type, FACE, double>::type, //
        typename traits::field_type<mesh_type, VOLUME, double>::type, //

        typename traits::field_type<mesh_type, VERTEX, Vec3>::type, //
        typename traits::field_type<mesh_type, EDGE, Vec3>::type, //
        typename traits::field_type<mesh_type, FACE, Vec3>::type, //
        typename traits::field_type<mesh_type, VOLUME, Vec3>::type  //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::mesh_type> //
        TestField<TF>::mesh = std::make_shared<typename TestField<TF>::mesh_type>();

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

