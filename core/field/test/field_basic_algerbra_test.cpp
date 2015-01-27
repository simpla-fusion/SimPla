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
#include "../../diff_geometry/dummy_mesh.h"

using namespace simpla;

typedef testing::Types< //

		Field<Domain<DummyMesh>, double> //
		, Field<Domain<DummyMesh>, nTuple<double, 3> > //
		, Field<Domain<DummyMesh>, nTuple<double, 3, 3> > //
		, Field<Domain<DummyMesh>, nTuple<std::complex<double>, 3> > //

> TypeParamList;
template<typename TF> std::shared_ptr<typename TestField<TF>::manifold_type> //
TestField<TF>::manifold = std::make_shared<DummyMesh>(10, 20);

INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

