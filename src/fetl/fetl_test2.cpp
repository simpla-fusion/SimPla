/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
#include "fetl_test2.h"

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

TestFETLParam<TMesh, Complex, VERTEX>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(FETL, TestFETLVecAlgegbra, ParamList);
