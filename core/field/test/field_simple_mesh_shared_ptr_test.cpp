/*
 * @file field_simple_mesh_shared_ptr_test.cpp
 *
 *  Created on: 2015-1-29
 *      Author: salmon
 */
#include <gtest/gtest.h>

#include "../../gtl/ntuple.h"
#include "../../mesh/simple_mesh.h"
#include "../field_shared_ptr.h"
#include "../field.h"
#include "field_basic_test.h"

using namespace simpla;

typedef Field<SimpleMesh, std::shared_ptr<nTuple<double, 3> > > vfield_type;

template<typename TField, int id>
struct TestFieldParam
{
	typedef TField field_type;

	static SimpleMesh mesh;
};

typedef _Field<SimpleMesh, std::shared_ptr<double>> field_type;

template<> SimpleMesh TestFieldParam<field_type, 0>::mesh = SimpleMesh();
template<> SimpleMesh TestFieldParam<field_type, 1>::mesh = SimpleMesh();

typedef testing::Types<

TestFieldParam<field_type, 0>,

TestFieldParam<field_type, 1>

> TypeParamList;
INSTANTIATE_TYPED_TEST_CASE_P(FIELD, TestField, TypeParamList);

