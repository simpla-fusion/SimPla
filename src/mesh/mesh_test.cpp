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

typedef RectMesh<> Mesh;

using namespace simpla;

template<typename TM, int IFORM, typename TV = Real> class Form
{
public:
	typedef TV value_type;
	typedef TM mesh_type;
	static constexpr int IForm = IFORM;
};

typedef testing::Types<Form<Mesh, VERTEX>, Form<Mesh, EDGE>, Form<Mesh, FACE>, Form<Mesh, VOLUME> > FormList;

INSTANTIATE_TYPED_TEST_CASE_P(Mesh, TestMesh, FormList);
