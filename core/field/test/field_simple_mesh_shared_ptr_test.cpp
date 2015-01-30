/*
 * @file field_simple_mesh_shared_ptr_test.cpp
 *
 *  Created on: 2015年1月29日
 *      Author: salmon
 */

#include "../../gtl/ntuple.h"
#include "../../diff_geometry/simple_mesh.h"
#include "../field_shared_ptr.h"
using namespace simpla;

typedef SimpleMesh mesh_type;
typedef std::shared_ptr<double> container_type;

#include "field_basic_test.h"

INSTANTIATE_TEST_CASE_P(FETLCartesian, TestFieldCase,
		testing::Values(
				SimpleMesh(nTuple<Real, 3>( { 0, 0, 0 }), nTuple<Real, 3>( { 1,
						1, 1 }), nTuple<size_t, 3>( { 10, 10, 10 }),
						nTuple<size_t, 3>( { 0, 0, 0 }))));
