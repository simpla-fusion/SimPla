/*
 * model_test.cpp
 *
 *  Created on: 2014年3月12日
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <random>
#include <string>
#include "../fetl/fetl.h"
#include "../fetl/save_field.h"

#include "model.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_euclidean.h"

using namespace simpla;

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

class TestModel: public testing::TestWithParam<

std::tuple<

typename TMesh::coordinates_type,

typename TMesh::coordinates_type,

nTuple<TMesh::NDIMS, size_t>

> >
{
protected:
	virtual void SetUp()
	{
		auto param = GetParam();

		xmin = std::get<0>(param);

		xmax = std::get<1>(param);

		dims = std::get<2>(param);

		mesh.SetExtents(xmin, xmax, dims);

		model = std::shared_ptr<model_type>(new model_type(mesh));

		auto extent = mesh.GetExtents();

		for (int i = 0; i < NDIMS; ++i)
		{
			dh[i] = (dims[i] > 1) ? (extent.second[i] - extent.first[i]) / dims[i] : 0;
		}

		points.emplace_back(coordinates_type( { 0.2 * xmax[0], 0.2 * xmax[1], 0.2 * xmin[2] }));

		points.emplace_back(coordinates_type( { 0.8 * xmax[0], 0.8 * xmax[1], 0.2 * xmin[2] }));

		GLOBAL_DATA_STREAM.OpenFile("MaterialTest");
		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	typedef TMesh mesh_type;
	typedef Real value_type;
	typedef mesh_type::iterator iterator;
	typedef mesh_type::coordinates_type coordinates_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;
	typedef Model<mesh_type> model_type;

	mesh_type mesh;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	static constexpr unsigned int IForm = VERTEX;

	nTuple<NDIMS, Real> xmin;

	nTuple<NDIMS, Real> xmax;

	nTuple<NDIMS, size_t> dims;

	std::shared_ptr<model_type> model;

	nTuple<NDIMS, Real> dh;

	std::vector<coordinates_type> points;

};

TEST_P(TestModel,ZeroForm )
{

//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	auto extent = mesh.GetExtents();
	CHECK(points);
	model->Set("Plasma", points);
	model->Update();

	CHECK(model->material_.size());

//	TZeroForm f(mesh);
//
//	f.Clear();
//
//	for (auto s : model->SelectByName(mesh.Select(IForm), "Plasma"))
//	{
//		f[s] = 1;
//	}
//	LOGGER << SAVE(f);
//
//	coordinates_type v0, v1, v2, v3;
//	for (int i = 0; i < NDIMS; ++i)
//	{
//		v0[i] = points[0][i] + dh[i];
//		v1[i] = points[1][i] - dh[i];
//
//		v2[i] = points[0][i] - dh[i] * 2;
//		v3[i] = points[1][i] + dh[i] * 2;
//	}
//	for (auto s : mesh.Select(IForm))
//	{
//		auto x = mesh.GetCoordinates(s);
//
//		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
//		        && (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
//		{
//			EXPECT_EQ(1,f[s] ) << (mesh.GetCoordinates(s));
//		}
//
//		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
//		        && (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
//		{
//			EXPECT_NE(1,f[s]) << (mesh.GetCoordinates(s));
//		}
//	}
//
//	points.emplace_back(coordinates_type( { 0.3 * extent.second[0], 0.6 * extent.second[1], 0.2 * extent.first[2] }));
//
//	model->Remove("Plasma", points);
//	model->Update();
//
//	f.Clear();
//
//	for (auto s : model->SelectByName(mesh.Select(IForm), "Plasma"))
//	{
//		f[s] = 1;
//	}
//
//	LOGGER << SAVE(f);
//
//	for (auto s : model->SelectInterface(mesh.Select(IForm), "Plasma", "NONE"))
//	{
//		f[s] = 10;
//	}
//
//	for (auto s : model->SelectInterface(mesh.Select(IForm), "Vacuum", "Plasma"))
//	{
//		f[s] = -10;
//	}
//	LOGGER << SAVE(f);

}

INSTANTIATE_TEST_CASE_P(FETL, TestModel,

testing::Combine(testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })  //
//        , nTuple<3, Real>( { -1.0, -2.0, -3.0 } )
        ),

testing::Values(

nTuple<3, Real>( { 1.0, 2.0, 3.0 })  //
//        , nTuple<3, Real>( { 2.0, 0.0, 2.0 }) //
//        , nTuple<3, Real>( { 2.0, 2.0, 0.0 }) //

        ),

testing::Values(nTuple<3, size_t>( { 12, 16, 10 }) //
//        , nTuple<3, size_t>( { 1, 10, 20 }) //
//        , nTuple<3, size_t>( { 17, 1, 17 }) //
//        , nTuple<3, size_t>( { 17, 17, 1 }) //

        )

        ));
