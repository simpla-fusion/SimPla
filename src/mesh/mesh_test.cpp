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

		17, 33, 65,

		17, 1, 1,

		1, 17, 1,

		1, 1, 17,

		1, 17, 17,

		17, 1, 17,

		17, 17, 1,

		17, 17, 17

		};
		mesh->SetExtents(xmin, xmax);

		mesh->SetDimensions(dims[ICASE % 100]);

	}

};

typedef testing::Types<

//TestMeshParam<Mesh , 0>,
//
//TestMeshParam<Mesh , 1>,

        TestMeshParam<TMesh, 0> //,

//TestMeshParam<Mesh , 5>,
//
//TestMeshParam<Mesh , 101>,
//
//TestMeshParam<Mesh , 103>,
//
//TestMeshParam<Mesh , 105>,
//
//
//TestMeshParam<Mesh  0>,
//
//TestMeshParam<Mesh , 100>,
//
//TestMeshParam<Mesh  100>
//
//TestMeshParam<Mesh , 0>,
//
//TestMeshParam<Mesh , 0>,
//
//TestMeshParam<Mesh , 0>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestMesh, ParamList);

