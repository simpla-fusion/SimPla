/*
 * mesh_test.h
 *
 *  Created on: 2014年3月25日
 *      Author: salmon
 */

#ifndef MESH_TEST_H_
#define MESH_TEST_H_

#include <gtest/gtest.h>

#include "../fetl/fetl.h"
#include "../io/data_stream.h"
#include "../utilities/log.h"

using namespace simpla;

template<typename TF>
class TestMesh: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);
		dims_list.emplace_back(nTuple<3, size_t>( { 17, 1, 1 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 1, 17, 1 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 1, 1, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 1, 17, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 17, 1, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 17, 17, 1 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 17, 17, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>( { 17, 33, 65 }));

	}
public:

	typedef typename TF::mesh_type mesh_type;
	typedef typename TF::value_type value_type;
	typedef typename mesh_type::index_type index_type;

	static constexpr int IForm = TF::IForm;
	std::vector<nTuple<3, size_t>> dims_list;

};

TYPED_TEST_CASE_P(TestMesh);
//TYPED_TEST_P(TestMesh, next){
//{
//	for(auto const & dims:TestFixture::dims_list)
//	{
//		typename TestFixture::mesh_type mesh;
//
//		mesh.SetDimensions(dims);
//
//		mesh.Update();
////		auto Range=mesh.GetRange(TestFixture::IForm);
////
////		CHECK_BIT(Range.begin()->d);
////
////		CHECK_BIT(Range.end()->d);
////
////		auto s=*Range.begin();
////
////		for (int i = 0; i < mesh.GetNumOfElements( TestFixture::IForm); ++i)
////		{
////			s=mesh.Next(s);
////
//////			CHECK_BIT(s.d );
////
////			CHECK_BIT(s.d );
////		}
//		size_t count = 0;
//
//		for(auto s:mesh.GetRange(TestFixture::IForm))
//		{
//			++count;
//		}
//
//		EXPECT_EQ(count, mesh.GetNumOfElements( TestFixture::IForm))<<mesh.GetDimensions();
//	}
//
//}}
TYPED_TEST_P(TestMesh, traversal){
{
	for(auto const & dims:TestFixture::dims_list)
	{
		typename TestFixture::mesh_type mesh;

		mesh.SetDimensions(dims);

		mesh.Update();

		size_t count = 0;

		for(auto s:mesh.GetRange(TestFixture::IForm ) )
		{
			++count;
		}

		EXPECT_EQ(count, mesh.GetNumOfElements( TestFixture::IForm))<<mesh.GetDimensions();

	}

}}
TYPED_TEST_P(TestMesh, partial_traversal){
{
	for(auto const & dims:TestFixture::dims_list)
	{
		typename TestFixture::mesh_type mesh;

		mesh.SetDimensions(dims);

		mesh.Update();

		int total=4;
		size_t count = 0;

		for (int s = 0; s < total; ++s)
		{
			for(auto s:mesh.GetRange(TestFixture::IForm,total,s ) )
			{
				++count;
			}
		}

		EXPECT_EQ(count, mesh.GetNumOfElements( TestFixture::IForm))<<mesh.GetDimensions();
	}

}}

REGISTER_TYPED_TEST_CASE_P(TestMesh, traversal, partial_traversal/**/);

//typedef testing::Types<RectMesh<>
////, CoRectMesh<Complex>
//> AllMeshTypes;
//
//template<typename TMesh>
//class TestMesh: public testing::Test
//{
//protected:
//	virtual void SetUp()
//	{
//		Logger::Verbose(10);
//	}
//public:
//
//	typedef TMesh mesh_type;
//
//	DEFINE_FIELDS(mesh_type)
//};
//
//TYPED_TEST_CASE(TestMesh, AllMeshTypes);
//
//TYPED_TEST(TestMesh,create_default){
//{
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	mesh_type mesh;
//
//	mesh.Update();
//
//	LOGGER<<mesh;
//
//}
//}
//TYPED_TEST(TestMesh,Create_parse_cfg){
//{
//	LuaObject cfg;
//	cfg.ParseString(
//
//			" Grid=                                                                       \n"
//			" {                                                                                            \n"
//			"   Type=\"CoRectMesh\",                                                                       \n"
//			"   ScalarType=\"Real\",                                                                  \n"
//			"   UnitSystem={Type=\"SI\"},                                                                  \n"
//			"   Topology=                                                                                  \n"
//			"   {                                                                                          \n"
//			"       Type=\"3DCoRectMesh\",                                                                 \n"
//			"       Dimensions={100,100,100}, -- number of grid, now only first dimension is valid            \n"
//			"       GhostWidth= {5,0,0},  -- width of ghost points  , if gw=0, coordinate is               \n"
//			"                               -- Periodic at this direction                                  \n"
//			"   },                                                                                         \n"
//			"   Geometry=                                                                                  \n"
//			"   {                                                                                          \n"
//			"       Type=\"Origin_DxDyDz\",                                                                \n"
//			"       Min={0.0,0.0,0.0},                                                                     \n"
//			"       Max={1.0,1.0,1.0},                                                                     \n"
//			"       dt=0.5*1.0/ (100.0-1.0)  -- time step                                                      \n"
//			"   }                                                                                          \n"
//			"}                                                                                             \n"
//
//	);
//
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	mesh_type mesh;
//
//	mesh.Deserialize( cfg["Grid"]);
//
//	LOGGER<<mesh;
//
//}
//}
//
//template<typename TMesh>
//class TestMeshFunctions: public testing::Test
//{
//protected:
//	virtual void SetUp()
//	{
//		Logger::Verbose(10);
//
//		mesh.SetDt(1.0);
//
//		nTuple<3, Real> xmin = { 0, 0, 0 };
//		nTuple<3, Real> xmax = { 1, 1, 1 };
//		mesh.SetExtent(xmin, xmax);
//
//		nTuple<3, size_t> dims = { 20, 20, 20 };
//		mesh.SetDimensions(dims);
//
//		mesh.Update();
//
//	}
//public:
//
//	typedef TMesh mesh_type;
//
//	DEFINE_FIELDS(mesh_type)
//
//	mesh_type mesh;
//
//};
//
//TYPED_TEST_CASE(TestMeshFunctions, AllMeshTypes);
//
//TYPED_TEST(TestMeshFunctions,add_tags){
//{
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	mesh_type & mesh=TestFixture::mesh;
//
//	LuaObject cfg;
//
//	cfg.ParseString(
//			" Media=                                                                 "
//			" {                                                                      "
//			"    {Type=\"Vacuum\",Range={{0.2,0,0},{0.8,0,0}},Op=\"Set\"},          "
//			"                                                                        "
//			"    {Type=\"Plasma\",                                                   "
//			"      Select=function(x,y,z)                                            "
//			"           return x>1.0 and x<2.0 ;                                     "
//			"         end                                                            "
//			"     ,Op=\"Set\"},                                                      "
//			" }                                                                      "
//	);
//
//	mesh.tags().Deserialize(cfg["Media"]);
//
//}}

//*******************************************************************************************************

//template<typename TMesh>
//class Test1DMesh: public testing::Test
//{
//protected:
//	virtual void SetUp()
//	{
//		Logger::Verbose(10);
//
//		mesh.dt_ = 1.0;
//		mesh.xmin_[0] = 0;
//		mesh.xmin_[1] = 0;
//		mesh.xmin_[2] = 0;
//		mesh.xmax_[0] = 1.0;
//		mesh.xmax_[1] = 1.0;
//		mesh.xmax_[2] = 1.0;
//		mesh.dims_[0] = 20;
//		mesh.dims_[1] = 1;
//		mesh.dims_[2] = 1;
//		mesh.dt_ = 1.0;
//
//		mesh.Update();
//
//		GLOBAL_DATA_STREAM.OpenFile("");
//
//	}
//public:
//
//	typedef TMesh mesh_type;
//
//	DEFINE_FIELDS(mesh_type)
//
//	mesh_type mesh;
//
//};
//TYPED_TEST_CASE(Test1DMesh, AllMeshTypes);
//
//TYPED_TEST(Test1DMesh,shift){
//{
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	mesh_type & mesh=TestFixture::mesh;
//
//	EXPECT_EQ(mesh.Shift(mesh.INC(0),105),6L);
//	EXPECT_EQ(mesh.Shift(mesh.INC(1),105),5L);
//	EXPECT_EQ(mesh.Shift(mesh.INC(2),105),5L);
//
//	EXPECT_EQ(mesh.Shift(mesh.DES(0),105),4L);
//	EXPECT_EQ(mesh.Shift(mesh.DES(1),105),5L);
//	EXPECT_EQ(mesh.Shift(mesh.DES(2),105),5L);
//
//	auto s= mesh.GetIndex(3,4,5);
//	EXPECT_EQ(mesh.Shift(mesh.DES(0),3,4,5),s-1);
//	EXPECT_EQ(mesh.Shift(mesh.DES(1),3,4,5),s);
//	EXPECT_EQ(mesh.Shift(mesh.DES(2),3,4,5),s);
//
//	EXPECT_EQ(mesh.Shift(mesh.INC(0),3,4,5),s+1);
//	EXPECT_EQ(mesh.Shift(mesh.INC(1),3,4,5),s);
//	EXPECT_EQ(mesh.Shift(mesh.INC(2),3,4,5),s);
//
//}}

#endif /* MESH_TEST_H_ */
