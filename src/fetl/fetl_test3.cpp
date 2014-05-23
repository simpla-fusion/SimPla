/*
 * fetl_test3.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */
#include "fetl_test.h"
#include "fetl_test3.h"
#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"
using namespace simpla;
typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

template<typename TV, int ICase>
struct TestFETLParam<TMesh, TV, ICase>
{
	typedef TMesh mesh_type;
	typedef TV value_type;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin[] =
		{

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		-1.0, -1.0, -1.0 };

		nTuple<3, Real> xmax[] =
		{

		1.0, 1.0, 1.0,

		2.0, 0.0, 0.0,

		0.0, 2.0, 0.0,

		0.0, 0.0, 2.0,

		0.0, 2.0, 2.0,

		2.0, 0.0, 2.0,

		2.0, 2.0, 0.0,

		1.0, 2.0, 3.0

		};

		constexpr int NUM_DIMS_TEST = 8;
		nTuple<3, size_t> dims[] =
		{

		16, 16, 16

		, 17, 1, 1

		, 1, 17, 1

		, 1, 1, 17

		, 1, 17, 17

		, 17, 1, 17

		, 17, 17, 1

		, 17, 17, 17

		};

		mesh->SetDimensions(dims[ICase % 10]);

		mesh->SetExtents(xmin[(ICase % 100) / 10], xmax[(ICase % 100) / 10]);

	}

	static void SetDefaultValue(value_type * v)
	{
		::SetDefaultValue(v);
	}
};

typedef testing::Types<

TestFETLParam<TMesh, Real, 0> //,
//
//TestFETLParam<TMesh, Real, 10>,
//
//TestFETLParam<TMesh, Real, 20>,
//
//TestFETLParam<TMesh, Real, 30>,
//
//TestFETLParam<TMesh, Real, 40>,
//
//TestFETLParam<TMesh, Real, 50>,
//
//TestFETLParam<TMesh, Real, 60>,
//
//
//TestFETLParam<TMesh, Real, 1>,
//
//TestFETLParam<TMesh, Real, 2>,
//
//TestFETLParam<TMesh, Real, 3>,
//
//TestFETLParam<TMesh, Real, 4>,
//
//TestFETLParam<TMesh, Real, 5>,
//
//TestFETLParam<TMesh, Real, 6>,
//
//TestFETLParam<TMesh, Real, 7>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestDiffCalculus, ParamList);
