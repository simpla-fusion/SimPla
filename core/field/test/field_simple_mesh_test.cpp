/*
 * @file field_simple_mesh_test.cpp
 *
 *  Created on: 2015年1月29日
 *      Author: salmon
 */

#include "../../gtl/ntuple.h"
#include "../../diff_geometry/simple_mesh.h"
using namespace simpla;

typedef _Field<SimpleMesh, std::shared_ptr<double>> f_type;

#include "field_test.h"

INSTANTIATE_TEST_CASE_P(FETLCartesian, TestFieldCase,
		testing::Values(SimpleMesh(nTuple<Real, 3>(
		{ 0, 0, 0 }), nTuple<Real, 3>(
		{ 1, 1, 1 }), nTuple<size_t, 3>(
		{ 10, 10, 10 }), nTuple<size_t, 3>(
		{ 0, 0, 0 }))));
