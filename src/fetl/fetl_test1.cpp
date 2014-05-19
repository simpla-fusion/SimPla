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
#include "../mesh/mesh.h"
#include "../mesh/geometry_euclidean.h"
typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

template<typename TV, int IFORM>
struct TestFETLParam<TMesh, TV, IFORM>
{
	typedef TMesh mesh_type;
	typedef TV value_type;
	static constexpr int IForm = IFORM;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin = { -1.0, -1.0, -1.0 };

		nTuple<3, Real> xmax = { 1.0, 1.0, 1.0 };

		nTuple<3, size_t> dims = { 16, 32, 67 };

		mesh->SetDimensions(dims);

		mesh->SetExtents(xmin, xmax);

	}

	static void SetDefaultValue(value_type * v)
	{
		::SetDefaultValue(v);
	}
};

typedef testing::Types<

TestFETLParam<TMesh, Real, VERTEX>,

TestFETLParam<TMesh, Real, EDGE>,

TestFETLParam<TMesh, Real, FACE>,

TestFETLParam<TMesh, Real, VOLUME>,

TestFETLParam<TMesh, Complex, VERTEX>,

TestFETLParam<TMesh, Complex, EDGE>,

TestFETLParam<TMesh, Complex, FACE>,

TestFETLParam<TMesh, Complex, VOLUME>,

TestFETLParam<TMesh, nTuple<3, Real>, VERTEX>,

TestFETLParam<TMesh, nTuple<3, Real>, EDGE>,

TestFETLParam<TMesh, nTuple<3, Real>, FACE>,

TestFETLParam<TMesh, nTuple<3, Real>, VOLUME>,

TestFETLParam<TMesh, nTuple<3, Complex>, VERTEX>,

TestFETLParam<TMesh, nTuple<3, Complex>, EDGE>,

TestFETLParam<TMesh, nTuple<3, Complex>, FACE>,

TestFETLParam<TMesh, nTuple<3, Complex>, VOLUME>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Real>>, VERTEX>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Real>>, EDGE>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Real>>, FACE>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Real>>, VOLUME>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Complex>>, VERTEX>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Complex>>, EDGE>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Complex>>, FACE>,

TestFETLParam<TMesh, nTuple<3, nTuple<3, Complex>>, VOLUME>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETL, ParamList);
