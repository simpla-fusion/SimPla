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
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"

using namespace simpla;

template<typename TInt>
class TestModel: public testing::Test
{
protected:
	virtual void SetUp()
	{

		mesh.SetExtents(xmin, xmax, dims);

		model = std::shared_ptr<model_type>(new model_type(mesh));

		auto extent = mesh.GetExtents();

		for (int i = 0; i < NDIMS; ++i)
		{
			dh[i] = (dims[i] > 1) ? (extent.second[i] - extent.first[i]) / dims[i] : 0;
		}

		points.emplace_back(coordinates_type( { 0.2 * xmax[0], 0.2 * xmax[1], 0.2 * xmin[2] }));

		points.emplace_back(coordinates_type( { 0.8 * xmax[0], 0.8 * xmax[1], 0.8 * xmax[2] }));

//		GLOBAL_DATA_STREAM.OpenFile("MaterialTest");
//		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	typedef Mesh<CartesianGeometry<UniformArray>> mesh_type;
	typedef Real value_type;
	typedef mesh_type::iterator iterator;
	typedef mesh_type::coordinates_type coordinates_type;
	typedef Model<mesh_type> model_type;

	mesh_type mesh;
	static constexpr unsigned int IForm = TInt::value;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	nTuple<NDIMS, Real> xmin = { 0.0, 0.0, 0.0, };

	nTuple<NDIMS, Real> xmax = { 1.0, 2.0, 3.0 };

	nTuple<NDIMS, size_t> dims = { 50, 60, 10 };

	std::shared_ptr<model_type> model;

	nTuple<NDIMS, Real> dh;

	std::vector<coordinates_type> points;

};
TYPED_TEST_CASE_P(TestModel);

TYPED_TEST_P(TestModel,SelectByRectangle ){
{

	auto   & model= *TestFixture::model;
	auto const & mesh= TestFixture::mesh;
	static constexpr unsigned int IForm=TestFixture::IForm;

	typename TestFixture::coordinates_type v0, v1, v2, v3;

	for (int i = 0; i < TestFixture::NDIMS; ++i)
	{
		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];

		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
	}

	auto f = mesh.template make_field<IForm,Real>( );

	f.clear();

	for (auto s : model.SelectByRectangle( TestFixture::IForm, TestFixture::points[0],TestFixture::points[1]))
	{
		f[s] = 1;
		auto x = mesh.GetCoordinates(s);

		ASSERT_TRUE (

				( (v2[0] - x[0]) * (x[0] - v3[0]) >= 0) &&

				( (v2[1] - x[1]) * (x[1] - v3[1]) >= 0) &&

				( (v2[2] - x[2]) * (x[2] - v3[2]) >= 0)

		)

		;
	}
//	LOGGER << SAVE(f);
	for (int i = 0; i < TestFixture::NDIMS; ++i)
	{
		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];

		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
	}
	for (auto s : model.Select(TestFixture::IForm))
	{
		auto x = mesh.GetCoordinates(s);

		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
		{
			ASSERT_EQ(1,f[s] ) << ( mesh.GetCoordinates(s));
		}

		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
				&& (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
		{
			ASSERT_NE(1,f[s]) << ( mesh.GetCoordinates(s));
		}
	}

//	LOGGER << SAVE(f);

}}

TYPED_TEST_P(TestModel,SelectByPolylines ){
{

	auto   & model= *TestFixture::model;
	auto const & mesh= TestFixture::mesh;
	static constexpr unsigned int IForm=TestFixture::IForm;

	auto f = mesh.template make_field<IForm,Real>( );

	f.clear();
	typename TestFixture::coordinates_type v0, v1, v2, v3;
	for (int i = 0; i < TestFixture::NDIMS; ++i)
	{
		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];

		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
	}
	for (auto s : model.SelectByPolylines( TestFixture::IForm, TestFixture::points))
	{
		f[s] = 1;
		auto x = mesh.GetCoordinates(s);

		ASSERT_TRUE (

				( (v2[0] - x[0]) * (x[0] - v3[0]) >= 0) &&

				( (v2[1] - x[1]) * (x[1] - v3[1]) >= 0) &&

				( (v2[2] - x[2]) * (x[2] - v3[2]) >= 0)

		)

		;
	}
//	LOGGER << SAVE(f);

}}

TYPED_TEST_P(TestModel,SelectByMaterial ){
{

	auto & model= *TestFixture::model;
	auto const & mesh= TestFixture::mesh;
	static constexpr unsigned int IForm=TestFixture::IForm;

	auto f = mesh.template make_field<IForm,Real>( );

	model.Set( model.SelectByPoints(VERTEX, TestFixture::points), "Vacuum");

	f.clear();

	for (auto s : model.SelectByMaterial(TestFixture::IForm, "Vacuum"))
	{
		f[s] = 1;
	}
//	LOGGER << SAVE(f);

	typename TestFixture::coordinates_type v0, v1, v2, v3;
	for (int i = 0; i < TestFixture::NDIMS; ++i)
	{
		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];

		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
	}
	for (auto s : model.Select(TestFixture::IForm))
	{
		auto x = mesh.GetCoordinates(s);

		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
		{
			ASSERT_EQ(1,f[s] ) << ( mesh.GetCoordinates(s));
		}

		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
				&& (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
		{
			ASSERT_NE(1,f[s]) << ( mesh.GetCoordinates(s));
		}
	}

	auto extent = mesh.GetExtents();

	TestFixture::points.emplace_back(typename TestFixture::coordinates_type(
					{	0.3 * extent.second[0], 0.6 * extent.second[1], 0.2 * extent.first[2]}));

	model.Erase( model.SelectByPolylines(VERTEX, TestFixture::points));

	model.Set( model.SelectByPolylines(VERTEX, TestFixture::points), "Plasma");

	for (auto s : model.SelectByMaterial( TestFixture::IForm, "Plasma"))
	{
		f[s] = -1;
	}

	for (auto s : model.SelectInterface( TestFixture::IForm, "Plasma", "Vacuum"))
	{
		f[s] = 10;
	}

	for (auto s : model.SelectInterface( TestFixture::IForm, "Vacuum", "NONE"))
	{
		f[s] = -10;
	}
//	LOGGER << SAVE(f);

}}

REGISTER_TYPED_TEST_CASE_P(TestModel, SelectByRectangle, SelectByPolylines, SelectByMaterial);

typedef testing::Types<Int2Type<VERTEX>, Int2Type<EDGE>, Int2Type<FACE>, Int2Type<VOLUME> > ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestModel, ParamList);
