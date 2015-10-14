<<<<<<< HEAD
/**
 * @file field_basic_algebra_test.cpp
=======
/*
 * field_basic_algerbra_test.cpp
>>>>>>> origin/master
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>

<<<<<<< HEAD

#include "../../field/field.h"
#include "../../field/field_dense.h"
#include "../../geometry/cs_cartesian.h"
#include "../../mesh/mock_mesh.h"
#include "../../mesh/domain.h"

=======
#include "field.h"
>>>>>>> origin/master
#include "field_basic_algerbra_test.h"
#include "../../diff_geometry/domain.h"
#include "../../diff_geometry/simple_mesh.h"

using namespace simpla;

<<<<<<< HEAD
typedef geometry::coordinate_system::Cartesian<3, 2> cs_type;

typedef MockMesh<cs_type> mesh_type;

=======
>>>>>>> origin/master
typedef testing::Types< //

		Field<Domain<SimpleMesh>, double> //
		, Field<Domain<SimpleMesh>, nTuple<double, 3> > //
		, Field<Domain<SimpleMesh>, nTuple<double, 3, 3> > //
		, Field<Domain<SimpleMesh>, nTuple<std::complex<double>, 3> > //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::manifold_type> //
TestField<TF>::mesh = std::make_shared<SimpleMesh>(10, 20);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

