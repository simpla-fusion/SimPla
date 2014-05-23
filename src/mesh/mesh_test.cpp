/*
 * mesh_test.cpp
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include "mesh_test.h"

#include "octree_forest.h"
#include "mesh_rectangle.h"
#include "geometry_euclidean.h"

using namespace simpla;

template<typename TM, int CASE> struct TestMeshParam;

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

typedef testing::Types<TMesh> RangeParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestRange, RangeParamList);

template<int CASE>
struct TestMeshParam<TMesh, CASE>
{

	static constexpr int ICASE = CASE % 100;
	static constexpr int IForm = CASE / 100;
	typedef TMesh mesh_type;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin = { 0, 0.0, 0 };

		nTuple<3, Real> xmax = { 1.0, 10, 1.0 };

		nTuple<3, size_t> dims[] = {

		1, 1, 1,

		15, 1, 1,

		1, 17, 1,

		1, 1, 19,

		1, 17, 18,

		19, 1, 17,

		17, 5, 1,

		17, 9, 7

		};
		mesh->SetExtents(xmin, xmax);

		mesh->SetDimensions(dims[ICASE % 100]);

	}

};

typedef testing::Types<

//TestMeshParam<TMesh, 0>,
//
//TestMeshParam<TMesh, 1>,
//
//TestMeshParam<TMesh, 2>,
//
//TestMeshParam<TMesh, 3>,
//
//TestMeshParam<TMesh, 4>,
//
//TestMeshParam<TMesh, 5>,
//
//TestMeshParam<TMesh, 6>,
//
//TestMeshParam<TMesh, 101>,
//
//TestMeshParam<TMesh, 102>,
//
//TestMeshParam<TMesh, 103>,
//
//TestMeshParam<TMesh, 104>,
//
//TestMeshParam<TMesh, 105>,
//
//TestMeshParam<TMesh, 106>,
//
//TestMeshParam<TMesh, 201>,
//
//TestMeshParam<TMesh, 202>,
//
//TestMeshParam<TMesh, 203>,
//
//TestMeshParam<TMesh, 204>,
//
//TestMeshParam<TMesh, 205>,
//
//TestMeshParam<TMesh, 206>,
//
//TestMeshParam<TMesh, 100>,
//
//TestMeshParam<TMesh, 200>,
////
        TestMeshParam<TMesh, 0>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestMesh, ParamList);

