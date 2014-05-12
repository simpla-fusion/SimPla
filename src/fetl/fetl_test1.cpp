/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */
#include "fetl_test1.h"
#include "fetl_test.h"
#include "fetl.h"
#include "../mesh/octree_forest.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/geometry_cylindrical.h"
#include "../mesh/geometry_euclidean.h"

typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;

template<typename TV, int IFORM>
struct TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, TV, IFORM>
{
	typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;
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

TestFETLParam<mesh_type, Real, VERTEX>,

TestFETLParam<mesh_type, Real, EDGE>,

TestFETLParam<mesh_type, Real, FACE>,

TestFETLParam<mesh_type, Real, VOLUME>,

TestFETLParam<mesh_type, Complex, VERTEX>,

TestFETLParam<mesh_type, Complex, EDGE>,

TestFETLParam<mesh_type, Complex, FACE>,

TestFETLParam<mesh_type, Complex, VOLUME>,

TestFETLParam<mesh_type, nTuple<3, Real>, VERTEX>,

TestFETLParam<mesh_type, nTuple<3, Real>, EDGE>,

TestFETLParam<mesh_type, nTuple<3, Real>, FACE>,

TestFETLParam<mesh_type, nTuple<3, Real>, VOLUME>,

TestFETLParam<mesh_type, nTuple<3, Complex>, VERTEX>,

TestFETLParam<mesh_type, nTuple<3, Complex>, EDGE>,

TestFETLParam<mesh_type, nTuple<3, Complex>, FACE>,

TestFETLParam<mesh_type, nTuple<3, Complex>, VOLUME>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Real>>, VERTEX>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Real>>, EDGE>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Real>>, FACE>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Real>>, VOLUME>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Complex>>, VERTEX>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Complex>>, EDGE>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Complex>>, FACE>,

TestFETLParam<mesh_type, nTuple<3, nTuple<3, Complex>>, VOLUME>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETL, ParamList);
