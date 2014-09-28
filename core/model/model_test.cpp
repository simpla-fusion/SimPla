/*
 * model_test.cpp
 *
 *  created on: 2014-3-12
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <random>
#include <string>
#include "../fetl/fetl.h"
#include "../fetl/field.h"
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

		xmin = nTuple<NDIMS, Real>( { 0.0, 0.0, 0.0 });

		xmax = nTuple<NDIMS, Real>( { 1.0, 2.0, 3.0 });

		dims = nTuple<NDIMS, size_t>( { 50, 60, 10 });

		model.set_dimensions(dims);

		model.set_extents(xmin, xmax);

		model.update();

		auto extent = model.get_extents();

		for (int i = 0; i < NDIMS; ++i)
		{
			dh[i] = (dims[i] > 1) ? (extent.second[i] - extent.first[i]) / dims[i] : 0;
		}

		points.emplace_back(coordinates_type( { 0.2 * xmax[0], 0.2 * xmax[1], 0.2 * xmin[2] }));

		points.emplace_back(coordinates_type( { 0.8 * xmax[0], 0.8 * xmax[1], 0.8 * xmax[2] }));

		LOGGER.set_stdout_visable_level(12);
//		GLOBAL_DATA_STREAM.cd("MaterialTest.h5:/");
	}
public:

	typedef Mesh<CartesianCoordinates<SurturedMesh>> mesh_type;
	typedef Real value_type;
	typedef mesh_type::iterator iterator;
	typedef mesh_type::coordinates_type coordinates_type;
	typedef Model<mesh_type> model_type;

	static constexpr unsigned int IForm = TInt::value;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	nTuple<NDIMS, Real> xmin/* = { 0.0, 0.0, 0.0, }*/;

	nTuple<NDIMS, Real> xmax/* = { 1.0, 2.0, 3.0 }*/;

	nTuple<NDIMS, size_t> dims/* = { 50, 60, 10 }*/;

	model_type model;

	nTuple<NDIMS, Real> dh;

	std::vector<coordinates_type> points;

};
TYPED_TEST_CASE_P(TestModel);

TYPED_TEST_P(TestModel,SelectByNGP){
{
	auto & model= TestFixture::model;

	typedef typename TestFixture::mesh_type mesh_type;

	typename TestFixture::coordinates_type min,max,x;

	std::tie(min,max)=model.get_extents();

	x=min*0.7+max*0.3;

	typename mesh_type::compact_index_type dest;

	std::tie(dest,std::ignore)=model.coordinates_global_to_local(x);

	static constexpr unsigned int IForm=TestFixture::IForm;

	auto range=model.SelectByNGP( TestFixture::IForm, x);

	size_t count =0;

	for(auto s :range)
	{
		EXPECT_EQ( mesh_type::get_cell_index(s),mesh_type::get_cell_index(dest));
		++count;
	}

	LOGGER<<count;

	EXPECT_EQ(count,mesh_type::get_num_of_comp_per_cell(IForm));
//
//	x= min-100;
//
//	std::tie(dest,std::ignore)=model.coordinates_global_to_local(x);
//
//	auto range2=model.SelectByNGP( TestFixture::IForm, x);
//
//	count =0;
//
//	for(auto s :range2)
//	{
//		EXPECT_EQ( mesh_type::get_cell_index(s),mesh_type::get_cell_index(dest));
//		++count;
//	}
//	EXPECT_EQ(count,mesh_type::get_num_of_comp_per_cell(IForm));

}
}

TYPED_TEST_P(TestModel,SelectByRectangle ){
{

	auto & model= TestFixture::model;

	static constexpr unsigned int IForm=TestFixture::IForm;

	typename TestFixture::coordinates_type v0, v1, v2, v3;

	for (int i = 0; i < TestFixture::NDIMS; ++i)
	{
		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];

		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
	}

	auto f = model.template make_field<IForm,Real>( );

	f.clear();

	auto r=model.SelectByRectangle( TestFixture::IForm, TestFixture::points[0],TestFixture::points[1]);

	auto it=std::begin(r);
	auto ie=std::end(r);
	for (; it!=ie;++it)
	{
		auto s=*it;

		f[s] = 1;
		auto x = model.get_coordinates(s);

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
	for (auto s : model.select(TestFixture::IForm))
	{
		auto x = model.get_coordinates(s);

		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
		{
			ASSERT_EQ(1,f[s] ) << ( model.get_coordinates(s));
		}

		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
				&& (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
		{
			ASSERT_NE(1,f[s]) << ( model.get_coordinates(s));
		}
	}

//	LOGGER << SAVE(f);

}}

TYPED_TEST_P(TestModel,SelectByPolylines ){
{

	auto & model= TestFixture::model;

	static constexpr unsigned int IForm=TestFixture::IForm;

	auto f = model.template make_field<IForm,Real>( );

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
		auto x = model.get_coordinates(s);

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

	auto & model= TestFixture::model;
	static constexpr unsigned int IForm=TestFixture::IForm;

	auto f = model.template make_field<IForm,Real>( );

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
	for (auto s : model.select(TestFixture::IForm))
	{
		auto x = model.get_coordinates(s);

		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
		{
			ASSERT_EQ(1,f[s] ) << ( model.get_coordinates(s));
		}

		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
				&& (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
		{
			ASSERT_NE(1,f[s]) << ( model.get_coordinates(s));
		}
	}

	auto extent = model.get_extents();

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

REGISTER_TYPED_TEST_CASE_P(TestModel, SelectByRectangle, SelectByPolylines, SelectByMaterial, SelectByNGP);

typedef testing::Types<std::integral_constant<unsigned int, VERTEX>, std::integral_constant<unsigned int, EDGE>,
        std::integral_constant<unsigned int, FACE>, std::integral_constant<unsigned int, VOLUME> > ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestModel, ParamList);
