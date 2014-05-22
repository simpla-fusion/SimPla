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
#include "../utilities/log.h"

using namespace simpla;

template<typename TParam>
class TestMesh: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);
		TParam::SetUpMesh(&mesh);
//		mesh.Decompose(1,0);
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename mesh_type::index_type index_type;
	mesh_type mesh;
	static constexpr int IForm = TParam::IForm;

};

TYPED_TEST_CASE_P(TestMesh);

TYPED_TEST_P(TestMesh, index){
{
	typedef typename TestFixture::mesh_type mesh_type;

//	s.d=1UL<<3;
//	CHECK_BIT(s.d);
//	s=s.NextNode(); CHECK_BIT(s.d);
//	s=s.NextNode(); CHECK_BIT(s.d);
//	s=s.NextNode(); CHECK_BIT(s.d);
//	s=s.NextNode(); CHECK_BIT(s.d);
//	s=s.NextNode(); CHECK_BIT(s.d);
}
}
TYPED_TEST_P(TestMesh, iterator){
{

}
}

TYPED_TEST_P(TestMesh, range){
{

}
}

TYPED_TEST_P(TestMesh, traversal){
{

	auto & mesh=TestFixture::mesh;

	size_t count = 0;

	auto range=mesh.GetRange(TestFixture::IForm );

	CHECK_BIT( range.first );
	CHECK_BIT( range.second );
	CHECK_BIT(*range.begin() );
	CHECK_BIT(*range.end() );

	auto it= range.begin();

	CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);
	++it; CHECK_BIT(*it);

	for(auto s:mesh.GetRange(TestFixture::IForm ) )
	{
		++count;
	}

	EXPECT_EQ(count, mesh.GetNumOfElements( TestFixture::IForm))<<mesh.GetDimensions();

}
}
TYPED_TEST_P(TestMesh, partial_traversal){
{

//	auto & mesh=TestFixture::mesh;
//
//	int total=4;
//	size_t count = 0;
//	auto range=mesh.GetRange(TestFixture::IForm);
//
//	auto dims= mesh.GetLocalDimensions();
//
//	auto start= range.first;
//
//	int data[dims[0]][dims[1]][dims[2]];
//
//	for (int i = 0; i < dims[0]; ++i)
//	{	for (int j = 0; j < dims[1]; ++j)
//		{
//			for (int k = 0; k < dims[2]; ++k)
//			{
//				data[i][j][k] =0;
//			}
//		}
//	}
////
////	for (int sub = 0; sub < total; ++sub)
////	{
////
////		for(auto s:range.Split(total,sub))
////		{
////			++count;
////////			CHECK(((s[0]-b[0])>>4)-dims[0]);
////////			CHECK(((s[1]-b[1])>>4)-dims[1]);
////////			CHECK(((s[2]-b[2])>>4)-dims[2]);
//////			EXPECT_LE((s[0]-start[0])>>4,dims[0]);
//////			EXPECT_LE((s[1]-start[1])>>4,dims[1]);
//////			EXPECT_LE((s[2]-start[2])>>4,dims[2]);
//////
//////			CHECK("") <<((s[0]-start[0])>>4)<< ","<<
//////			((s[1]-start[1])>>4)<< ","<<
//////			((s[2]-start[2])>>4 )
//////			<<std::endl;
//////
//////			data[(s[0]-start[0])>>4][(s[1]-start[1])>>4][(s[2]-start[2])>>4]=sub;
////		}
////	}
//
//	for (int i = 0; i < dims[0]; ++i)
//	{	for (int j = 0; j < dims[1]; ++j)
//		{
//			for (int k = 0; k < dims[2]; ++k)
//			{
//				std::cout<< data[i][j][k]<<" ";
//			}
//			std::cout<<std::endl;
//		}
//		std::cout<<"==========="<<std::endl;
//	}
//	EXPECT_EQ(count, mesh.GetNumOfElements( TestFixture::IForm))<<mesh.GetDimensions();

}}

//TYPED_TEST_P(TestMesh,scatter ){
//{
//	typedef typename TestFixture::mesh_type mesh_type;
//
//	auto & mesh=TestFixture::mesh;
//
//	Field<mesh_type,VERTEX,Real> n(mesh);
//	Field<mesh_type,EDGE,Real> J(mesh);
//
//	n.Clear();
//	J.Clear();
//
//	nTuple<3,Real> x=
//	{	-0.01,-0.01,-0.01};
//	nTuple<3,Real> v=
//	{	1,2,3};
//
//	mesh.Scatter(x,1.0,&n);
//	mesh.Scatter(x,v,&J);
//	for(auto const & v: n)
//	{
//		std::cout<<" "<<v;
//	}
//	std::cout<<std::endl;
//	for(auto const & v: J)
//	{
//		std::cout<<" "<<v;
//	}
//
//	std::cout<<std::endl;
//}}
//
//TYPED_TEST_P(TestMesh,gather){
//
//}
REGISTER_TYPED_TEST_CASE_P(TestMesh, index, iterator, range, traversal, partial_traversal/*, scatter, gather*/);

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
