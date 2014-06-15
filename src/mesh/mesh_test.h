/*
 * mesh_test.h
 *
 *  Created on: 2014年3月25日
 *      Author: salmon
 */

#ifndef MESH_TEST_H_
#define MESH_TEST_H_

#include <gtest/gtest.h>
#include "mesh.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../parallel/message_comm.h"

using namespace simpla;

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

class TestMesh: public testing::TestWithParam<

std::tuple<

typename TMesh::coordinates_type,

typename TMesh::coordinates_type,

nTuple<TMesh::NDIMS, size_t>

> >
{
protected:
	virtual void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

		auto param = GetParam();

		xmin=std::get<0>(param);

		xmax=std::get<1>(param);

		dims=std::get<2>(param);

		mesh.SetExtents(xmin,xmax,dims);

	}
public:
	typedef TMesh mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::range range;
	typedef typename mesh_type::iterator iterator;
	unsigned int NDIMS=TMesh::NDIMS;

	mesh_type mesh;

	std::vector<typename TMesh::compact_index_type> shift =
	{
		0UL,
		TMesh::_DI>>1,TMesh::_DJ>>1,TMesh::_DK>>1,
		(TMesh::_DJ|TMesh::_DK)>>1,(TMesh::_DK|TMesh::_DI)>>1,(TMesh::_DI|TMesh::_DJ)>>1,
		TMesh::_DA>>1
	};
	std::vector<unsigned int> iforms =
	{	VERTEX, EDGE, FACE, VOLUME};

	typename TMesh::coordinates_type xmin,xmax;
	nTuple<TMesh::NDIMS, index_type> dims;
};

TEST_P(TestMesh, ForAll)
{

	for (auto const & s : iforms)
	{
		nTuple<3, index_type> begin = { 0, 0, 0 };

		range r(s, begin, dims);

		size_t size = 1;

		for (int i = 0; i < NDIMS; ++i)
		{
			size *= dims[i];
		}

		size_t number = 0;

		std::vector<size_t> data;

		for (auto a : r)
		{
			data.push_back(number);
			++number;
		}

		CHECK(data);

		if (s == VERTEX || s == VOLUME)
		{
			EXPECT_EQ(data.size(), size);
		}
		else
		{
			EXPECT_EQ(data.size(), size * 3);
		}
	}
}
TEST_P(TestMesh, Split)
{

	for (auto const & s : iforms)
	{

		nTuple<3, index_type> begin = { 1, 3, 5 };

		nTuple<3, index_type> count = dims;

		range r(s, begin, count);

		size_t total = 4;

		std::vector<size_t> data;

		for (int sub = 0; sub < total; ++sub)
			for (auto a : r.Split(total, sub))
			{
				data.push_back(sub);
			}

		size_t size = 1;

		for (int i = 0; i < NDIMS; ++i)
		{
			size *= dims[i];
		}

		if (s == VERTEX || s == VOLUME)
		{
			EXPECT_EQ(data.size(), size);
		}
		else
		{
			EXPECT_EQ(data.size(), size * 3);
		}
	}

	//	CHECK(mesh.global_start_);
	//	CHECK(mesh.global_end_);
	//	CHECK_BIT( range.first );
	//	CHECK_BIT( range.second );
	//	CHECK_BIT( range.begin()->self_ );
	//	CHECK_BIT( range.end()->self_ );
	//	auto it= range.begin();
	//
	//	CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//	++it; CHECK_BIT(it->self_);
	//

}

TEST_P(TestMesh, coordinates)
{

	auto extents = mesh.GetExtents();

	auto range0 = mesh.Select(VERTEX);
	auto range1 = mesh.Select(EDGE);
	auto range2 = mesh.Select(FACE);
	auto range3 = mesh.Select(VOLUME);

	auto it = range1.begin();

	auto x = extents.second;
	for (int i = 0; i < NDIMS; ++i)
	{
		if (dims[i] <= 1 || xmax[i] <= xmin[i])
			x[i] = xmin[i];
	}
	EXPECT_EQ(x, mesh.GetCoordinates(*range0.rbegin()));

	EXPECT_DOUBLE_EQ(mesh.Volume(*range0.begin()) * mesh.Volume(*range3.begin()),
	        mesh.Volume(*range1.begin()) * mesh.Volume(*range2.begin()));

	EXPECT_DOUBLE_EQ(mesh.Volume(*range0.begin()), mesh.DualVolume(*range3.begin()));
	EXPECT_DOUBLE_EQ(mesh.Volume(*range1.begin()), mesh.DualVolume(*range2.begin()));

	it = range1.begin();
	EXPECT_EQ(mesh.ComponentNum(it.self_), 0);
	++it;
	EXPECT_EQ(mesh.ComponentNum(it.self_), 1);
	++it;
	EXPECT_EQ(mesh.ComponentNum(it.self_), 2);
	it = range2.begin();
	EXPECT_EQ(mesh.ComponentNum(it.self_), 0);
	++it;
	EXPECT_EQ(mesh.ComponentNum(it.self_), 1);
	++it;
	EXPECT_EQ(mesh.ComponentNum(it.self_), 2);

}

TEST_P(TestMesh, select )
{
	auto r = mesh.Select(VERTEX);
	size_t count = 0;
	for (auto a : r)
	{
		++count;
	}
	CHECK(count);
	EXPECT_EQ(count, dims[0] * dims[1] * dims[2]);
//	auto extent = mesh.GetExtents();
//	auto xmin = extent.first;
//	auto xmax = extent.second;
//
//	nTuple<3, size_t> start = { 0, 0, 0 };
//	CHECK(dims);
//
//	EXPECT_EQ(mesh.Decompact(*mesh.CoordinatesGlobalToLocal(&xmin)) - (mesh.global_start_ << mesh.D_FP_POS), start);
//	EXPECT_EQ(mesh.Decompact(*mesh.CoordinatesGlobalToLocal(&xmax)) - (mesh.global_start_ << mesh.D_FP_POS),
//	        dims << mesh.D_FP_POS);
}

TEST_P(TestMesh, volume)
{

	auto extents = mesh.GetExtents();

	auto range0 = mesh.Select(VERTEX);
	auto range1 = mesh.Select(EDGE);
	auto range2 = mesh.Select(FACE);
	auto range3 = mesh.Select(VOLUME);

	auto s = *range1.begin();
	auto X = mesh.topology_type::DeltaIndex(0, s);
	auto Y = mesh.topology_type::DeltaIndex(1, s);
	auto Z = mesh.topology_type::DeltaIndex(2, s);

//	CHECK(mesh.geometry_type::Volume(s));
//	CHECK(mesh.geometry_type::Volume(s + X));
//	CHECK(mesh.geometry_type::Volume(s - X));
//	CHECK(mesh.geometry_type::Volume(s + Y));
//	CHECK(mesh.geometry_type::Volume(s - Y));
//	CHECK(mesh.geometry_type::Volume(s + Z));
//	CHECK(mesh.geometry_type::Volume(s - Z));
//
//	CHECK(mesh.geometry_type::DualVolume(s));
//	CHECK(mesh.geometry_type::DualVolume(s + X));
//	CHECK(mesh.geometry_type::DualVolume(s - X));
//	CHECK(mesh.geometry_type::DualVolume(s + Y));
//	CHECK(mesh.geometry_type::DualVolume(s - Y));
//	CHECK(mesh.geometry_type::DualVolume(s + Z));
//	CHECK(mesh.geometry_type::DualVolume(s - Z));

}
TEST_P(TestMesh, traversal)
{

	for (unsigned int i = 0; i < 4; ++i)
	{
		auto IForm = iforms[i];

		std::map<size_t, mesh_type::compact_index_type> data;

		auto range = mesh.Select(IForm);

		for (auto s : mesh.Select(IForm))
		{
			data[mesh.Hash(s)] = s;
		}

		EXPECT_EQ(data.size(), mesh.GetNumOfElements(IForm)) << mesh.GetDimensions();
		EXPECT_EQ(data.begin()->first, 0);
		EXPECT_EQ(data.rbegin()->first, mesh.GetNumOfElements(IForm) - 1);
	}

}
TEST_P(TestMesh, partial_traversal)
{
	for (unsigned int i = 0; i < 4; ++i)
	{
		auto IForm = iforms[i];
		int total = 4;

		std::map<size_t, int> data;

		auto r = mesh.Select(IForm);

		for (int sub = 0; sub < total; ++sub)
		{

			for (auto s : r.Split(total, sub))
			{
				data[mesh.Hash(s)] = sub;
			}
		}

		EXPECT_EQ(data.size(), mesh.GetNumOfElements(IForm)) << mesh.GetDimensions();
		EXPECT_EQ(data.begin()->first, 0);
		EXPECT_EQ(data.rbegin()->first, mesh.GetNumOfElements(IForm) - 1);
	}
}

//TEST_P(TestMesh,scatter )
//{
//
//	Field<mesh_type, VERTEX, Real> n(mesh);
//	Field<mesh_type, EDGE, Real> J(mesh);
//
//	n.Clear();
//	J.Clear();
//
//	nTuple<3, Real> x = { -0.01, -0.01, -0.01 };
//	nTuple<3, Real> v = { 1, 2, 3 };
//
//	for (auto const & v : n)
//	{
//		std::cout << " " << v;
//	}
//	std::cout << std::endl;
//	for (auto const & v : J)
//	{
//		std::cout << " " << v;
//	}
//	std::cout << std::endl;
//	mesh.Scatter(x, 1.0, &n);
//	mesh.Scatter(x, v, &J);
//	for (auto const & v : n)
//	{
//		std::cout << " " << v;
//	}
//	std::cout << std::endl;
//	for (auto const & v : J)
//	{
//		std::cout << " " << v;
//	}
//
//	std::cout << std::endl;
//
//}
//
//TEST_P(TestMesh,gather)
//{
//
//}

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
//
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
//
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
//
//
//	mesh_type & mesh=mesh;
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
//
//
//	mesh_type & mesh=mesh;
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
