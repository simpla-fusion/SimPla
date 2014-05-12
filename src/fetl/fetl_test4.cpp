/*
 * fetl_test4.cpp
 *
 *  Created on: 2014年3月11日
 *      Author: salmon
 */
#include "fetl_test.h"
#include "fetl_test4.h"

#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"

typedef RectMesh<OcForest, EuclideanGeometry> Mesh;

template<typename TV, int IFORM>
struct TestFETLParam<Mesh, TV, IFORM>
{
	typedef Mesh mesh_type;
	typedef TV value_type;
	static constexpr int IForm = IFORM;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin = { -1.0, -1.0, -1.0 };

		nTuple<3, Real> xmax = { 1.0, 1.0, 1.0 };

		nTuple<3, size_t> dims = { 16, 32, 67 };

		mesh->SetDimensions(dims);

		mesh->SetExtent(xmin, xmax);
	}

	static void SetDefaultValue(value_type * v)
	{
		::SetDefaultValue(v);
	}
};

typedef testing::Types<

TestFETLParam<Mesh, Real, VERTEX>,

TestFETLParam<Mesh, Complex, VERTEX>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecField, ParamList);

