/*
 * field_basic_algerbra_test.cpp
 *
 *  Created on: Oct 11, 2014
 *      Author: salmon
 */

#include <iostream>
#include <gtest/gtest.h>

#include "field.h"
#include "field_basic_algerbra_test.h"
#include "../../diff_geometry/domain.h"
#include "../../diff_geometry/simple_mesh.h"

using namespace simpla;

typedef testing::Types< //

		Field<Domain<SimpleMesh>, double> //
		, Field<Domain<SimpleMesh>, nTuple<double, 3> > //
		, Field<Domain<SimpleMesh>, nTuple<double, 3, 3> > //
		, Field<Domain<SimpleMesh>, nTuple<std::complex<double>, 3> > //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::manifold_type> //
TestField<TF>::manifold = std::make_shared<SimpleMesh>(10, 20);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

