/*
 * @file field_simple_mesh_map_test.cpp
 *
 *  Created on: 2015年1月29日
 *      Author: salmon
 */

#include <unordered_map>
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "../../diff_geometry/simple_mesh.h"
using namespace simpla;

struct hash_id
{
public:
	size_t operator()(typename SimpleMesh::id_type const& s) const
	{
		return s[0];
	}
};
typedef _Field<SimpleMesh,
		std::unordered_map<typename SimpleMesh::id_type, double,hash_id>> f_type;

#include "field_test.h"

INSTANTIATE_TEST_CASE_P(FETLCartesian, TestFieldCase,
		testing::Values(SimpleMesh(nTuple<Real, 3>(
		{ 0, 0, 0 }), nTuple<Real, 3>(
		{ 1, 1, 1 }), nTuple<size_t, 3>(
		{ 5, 4, 6 }), nTuple<size_t, 3>(
		{ 0, 0, 0 }))));
