/*
 * @file field_cartesian_map_test.cpp
 *
 *  Created on: 2015年1月29日
 *      Author: salmon
 */
#include <unordered_map>
#include "../../gtl/ntuple.h"
#include "../../diff_geometry/diff_scheme/fdm.h"
#include "../../diff_geometry/interpolator/interpolator.h"
#include "../../diff_geometry/geometry/cartesian.h"
#include "../../diff_geometry/topology/structured.h"
#include "../../diff_geometry/mesh.h"

#include "../field_map.h"

using namespace simpla;

typedef Manifold<VERTEX, CartesianCoordinate<RectMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> mesh_type;

typedef std::unordered_map<typename mesh_type::id_type, double> container_type;

typedef _Field<mesh_type, container_type> f_type;

auto geo = CartesianCoordinate<RectMesh, CARTESIAN_ZAXIS>::create(
		nTuple<Real, 3>( { 0, 0, 0 }), nTuple<Real, 3>( { 1, 1, 1 }),
		nTuple<size_t, 3>( { 5, 4, 6 }), nTuple<size_t, 3>( { 0, 0, 0 }));

#include "field_basic_test.h"
//#include "field_diff_calculus_test.h"

INSTANTIATE_TEST_CASE_P(FETLCartesian, TestFieldCase,
		testing::Values(mesh_type(*geo)));
