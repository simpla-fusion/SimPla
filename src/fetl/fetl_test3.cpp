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

template<typename TV, int ICase>
struct TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, TV, ICase>
{
	typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;
	typedef TV value_type;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin[] = {

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		0.0, 0.0, 0.0,

		-1.0, -1.0, -1.0 };

		nTuple<3, Real> xmax[] = {

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
		nTuple<3, size_t> dims[] = {

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

		mesh->SetExtent(xmin[(ICase % 100) / 10], xmax[(ICase % 100) / 10]);

	}

	static void SetDefaultValue(value_type * v)
	{
		::SetDefaultValue(v);
	}
};

typedef testing::Types<

TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 0> //,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 10>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 20>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 30>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 40>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 50>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 60>,

//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 1>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 2>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 3>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 4>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 5>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 6>,
//
//TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Real, 7>,

/*TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, Complex, 8>,

 TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, nTuple<3, Real>, 8>*/

> ParamList;
INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestDiffCalculus, ParamList);
