/*
 * mesh_test.h
 *
 *  Created on: 2014年3月25日
 *      Author: salmon
 */

#ifndef MESH_TEST_H_
#define MESH_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../parallel/message_comm.h"
#include "uniform_array.h"
#include "geometry_cartesian.h"
#include "mesh_rectangle.h"

using namespace simpla;

#ifndef TMESH
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"
#include "../mesh/mesh_rectangle.h"

typedef Mesh<CartesianGeometry<UniformArray, false>> TMesh;
#else
typedef TMESH TMesh;
#endif

class TestMesh: public testing::TestWithParam<
        std::tuple<typename TMesh::coordinates_type, typename TMesh::coordinates_type, nTuple<TMesh::NDIMS, size_t> > >
{
protected:
	virtual void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

		auto param = GetParam();

		xmin=std::get<0>(param);

		xmax=std::get<1>(param);

		dims=std::get<2>(param);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || xmax[i] <= xmin[i])
			{
				xmax[i] = xmin[i];
				dims[i] = 1;
			}
		}

		mesh.SetExtents(xmin,xmax,dims);

	}
public:
	typedef TMesh mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename TMesh::coordinates_type coordinates_type;

	unsigned int NDIMS=TMesh::NDIMS;

	mesh_type mesh;

	std::vector<unsigned int> iform_list =
	{	VERTEX, EDGE, FACE, VOLUME};
	coordinates_type xmin,xmax;
	nTuple<TMesh::NDIMS, index_type> dims;

};
TEST_P(TestMesh, compact_index_type)
{
	auto s = mesh.get_first_node_shift(VERTEX);
	EXPECT_EQ(0, mesh.NodeId(s));
	EXPECT_EQ(0, mesh.NodeId(mesh.Roate(s)));
	EXPECT_EQ(0, mesh.NodeId(mesh.InverseRoate(s)));
	EXPECT_EQ(0, mesh.ComponentNum(mesh.Roate(s)));
	EXPECT_EQ(0, mesh.ComponentNum(mesh.InverseRoate(s)));
	EXPECT_EQ(VERTEX, mesh.NodeId(s));
	EXPECT_EQ(VERTEX, mesh.IForm(mesh.Roate(s)));
	EXPECT_EQ(VERTEX, mesh.IForm(mesh.InverseRoate(s)));

	s = mesh.get_first_node_shift(VOLUME);
	EXPECT_EQ(7, mesh.NodeId(s));
	EXPECT_EQ(7, mesh.NodeId(mesh.Roate(s)));
	EXPECT_EQ(7, mesh.NodeId(mesh.InverseRoate(s)));
	EXPECT_EQ(0, mesh.ComponentNum(mesh.Roate(s)));
	EXPECT_EQ(0, mesh.ComponentNum(mesh.InverseRoate(s)));

	EXPECT_EQ(VOLUME, mesh.IForm(s));
	EXPECT_EQ(VOLUME, mesh.IForm(mesh.Roate(s)));
	EXPECT_EQ(VOLUME, mesh.IForm(mesh.InverseRoate(s)));

	s = mesh.get_first_node_shift(EDGE);
	EXPECT_EQ(4, mesh.NodeId(s));
	EXPECT_EQ(2, mesh.NodeId(mesh.Roate(s)));
	EXPECT_EQ(1, mesh.NodeId(mesh.Roate(mesh.Roate(s))));
	EXPECT_EQ(1, mesh.NodeId(mesh.InverseRoate(s)));
	EXPECT_EQ(2, mesh.NodeId(mesh.InverseRoate(mesh.InverseRoate(s))));

	EXPECT_EQ(0, mesh.ComponentNum(s));
	EXPECT_EQ(1, mesh.ComponentNum(mesh.Roate(s)));
	EXPECT_EQ(2, mesh.ComponentNum(mesh.Roate(mesh.Roate(s))));
	EXPECT_EQ(2, mesh.ComponentNum(mesh.InverseRoate(s)));
	EXPECT_EQ(1, mesh.ComponentNum(mesh.InverseRoate(mesh.InverseRoate(s))));

	EXPECT_EQ(EDGE, mesh.IForm(s));
	EXPECT_EQ(EDGE, mesh.IForm(mesh.Roate(s)));
	EXPECT_EQ(EDGE, mesh.IForm(mesh.Roate(mesh.Roate(s))));
	EXPECT_EQ(EDGE, mesh.IForm(mesh.InverseRoate(s)));
	EXPECT_EQ(EDGE, mesh.IForm(mesh.InverseRoate(mesh.InverseRoate(s))));

	EXPECT_EQ(3, mesh.NodeId(mesh.Dual(s)));
	EXPECT_EQ(5, mesh.NodeId(mesh.Dual(mesh.Roate(s))));
	EXPECT_EQ(6, mesh.NodeId(mesh.Dual(mesh.InverseRoate(s))));

	s = mesh.get_first_node_shift(FACE);
	EXPECT_EQ(3, mesh.NodeId(s));
	EXPECT_EQ(5, mesh.NodeId(mesh.Roate(s)));
	EXPECT_EQ(6, mesh.NodeId(mesh.Roate(mesh.Roate(s))));
	EXPECT_EQ(6, mesh.NodeId(mesh.InverseRoate(s)));
	EXPECT_EQ(5, mesh.NodeId(mesh.InverseRoate(mesh.InverseRoate(s))));

	EXPECT_EQ(0, mesh.ComponentNum(s));
	EXPECT_EQ(1, mesh.ComponentNum(mesh.Roate(s)));
	EXPECT_EQ(2, mesh.ComponentNum(mesh.Roate(mesh.Roate(s))));
	EXPECT_EQ(2, mesh.ComponentNum(mesh.InverseRoate(s)));
	EXPECT_EQ(1, mesh.ComponentNum(mesh.InverseRoate(mesh.InverseRoate(s))));

	EXPECT_EQ(FACE, mesh.IForm(s));
	EXPECT_EQ(FACE, mesh.IForm(mesh.Roate(s)));
	EXPECT_EQ(FACE, mesh.IForm(mesh.Roate(mesh.Roate(s))));
	EXPECT_EQ(FACE, mesh.IForm(mesh.InverseRoate(s)));
	EXPECT_EQ(FACE, mesh.IForm(mesh.InverseRoate(mesh.InverseRoate(s))));

	EXPECT_EQ(4, mesh.NodeId(mesh.Dual(s)));
	EXPECT_EQ(2, mesh.NodeId(mesh.Dual(mesh.Roate(s))));
	EXPECT_EQ(1, mesh.NodeId(mesh.Dual(mesh.InverseRoate(s))));

}
TEST_P(TestMesh, iterator)
{

	for (auto const & iform : iform_list)
	{

		range_type r = mesh.Select(iform);

		size_t size = 1;

		for (int i = 0; i < NDIMS; ++i)
		{
			size *= dims[i];
		}

		std::set<typename mesh_type::compact_index_type> data;

		for (auto a : r)
		{
			data.insert(a);
		}

		if (iform == VERTEX || iform == VOLUME)
		{
			EXPECT_EQ(size, data.size()) << iform;
		}
		else
		{
			EXPECT_EQ(size * 3, data.size()) << iform;
		}
	}
}

TEST_P(TestMesh, coordinates)
{

	auto extents = mesh.GetExtents();

	auto range0 = mesh.Select(VERTEX);
	auto range1 = mesh.Select(EDGE);
	auto range2 = mesh.Select(FACE);
	auto range3 = mesh.Select(VOLUME);

	auto it = begin(range1);

	typename mesh_type::coordinates_type x = 0.21235 * (extents.second - extents.first) + extents.first;
	auto idx = mesh.topology_type::CoordinatesToIndex(x);

	EXPECT_EQ(idx, mesh.topology_type::CoordinatesToIndex(mesh.topology_type::IndexToCoordinates(idx)));

	EXPECT_EQ(x, mesh.CoordinatesLocalToGlobal(mesh.CoordinatesGlobalToLocal(x, mesh.get_first_node_shift(VERTEX))));
	EXPECT_EQ(x, mesh.CoordinatesLocalToGlobal(mesh.CoordinatesGlobalToLocal(x, mesh.get_first_node_shift(EDGE))));
	EXPECT_EQ(x, mesh.CoordinatesLocalToGlobal(mesh.CoordinatesGlobalToLocal(x, mesh.get_first_node_shift(FACE))));
	EXPECT_EQ(x, mesh.CoordinatesLocalToGlobal(mesh.CoordinatesGlobalToLocal(x, mesh.get_first_node_shift(VOLUME))));

}

TEST_P(TestMesh, volume)
{
	auto range0 = mesh.Select(VERTEX);
	auto range1 = mesh.Select(EDGE);
	auto range2 = mesh.Select(FACE);
	auto range3 = mesh.Select(VOLUME);

	EXPECT_DOUBLE_EQ(mesh.Volume(*begin(range0)) * mesh.Volume(*begin(range3)),
	        mesh.Volume(*begin(range1)) * mesh.Volume(*begin(range2)));

	EXPECT_DOUBLE_EQ(mesh.Volume(*begin(range0)), mesh.DualVolume(*begin(range3)));
	EXPECT_DOUBLE_EQ(mesh.Volume(*begin(range1)), mesh.DualVolume(*begin(range2)));

	auto extents = mesh.GetExtents();

	auto s = *begin(range1);

	EXPECT_DOUBLE_EQ(1.0, mesh.Volume(s + mesh.DeltaIndex(s)));
	EXPECT_DOUBLE_EQ(1.0, mesh.Volume(s - mesh.DeltaIndex(s)));

//	auto X = mesh.topology_type::DI(0, s);
//	auto Y = mesh.topology_type::DI(1, s);
//	auto Z = mesh.topology_type::DI(2, s);
//
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

TEST_P(TestMesh, hash)
{

	for (auto iform : iform_list)
	{

		std::map<size_t, mesh_type::compact_index_type> data;

		auto range = mesh.Select(iform);

		for (auto s : mesh.Select(iform))
		{
			data[mesh.Hash(s)] = s;
		}

		EXPECT_EQ(data.size(), mesh.GetNumOfElements(iform)) << mesh.GetDimensions();
		EXPECT_EQ(data.begin()->first, 0);
		EXPECT_EQ(data.rbegin()->first, mesh.GetNumOfElements(iform) - 1);
	}

}

TEST_P(TestMesh, Split)
{

//	for (auto const & iform : iform_list)

	unsigned int iform = VERTEX;
	{

		nTuple<3, index_type> begin = { 0, 0, 0 };

		nTuple<3, index_type> end = dims;

		auto r = mesh.make_range(begin, end, mesh.get_first_node_shift(iform));

		size_t total = 4;

		std::set<typename mesh_type::compact_index_type> data;

		for (int sub = 0; sub < total; ++sub)
			for (auto const & a : Split(r, total, sub))
			{
				data.insert(a);
			}

		size_t size = 1;

		for (int i = 0; i < NDIMS; ++i)
		{
			size *= dims[i];
		}

		if (iform == VERTEX || iform == VOLUME)
		{
			EXPECT_EQ(data.size(), size);
		}
		else
		{
			EXPECT_EQ(data.size(), size * 3);
		}
	}

}
TEST_P(TestMesh, partial_traversal)
{
	for (auto iform : iform_list)
	{
		int total = 4;

		std::map<size_t, int> data;

		auto r = mesh.Select(iform);

		for (int sub = 0; sub < total; ++sub)
		{

			for (auto s : Split(r, total, sub))
			{
				data[mesh.Hash(s)] = sub;
			}
		}

		EXPECT_EQ(data.size(), mesh.GetNumOfElements(iform)) << mesh.GetDimensions();
		EXPECT_EQ(data.begin()->first, 0);
		EXPECT_EQ(data.rbegin()->first, mesh.GetNumOfElements(iform) - 1);
	}
}

TEST_P(TestMesh, select )
{
	auto r = mesh.Select(VERTEX);

	std::set<typename mesh_type::compact_index_type> data;

	for (auto s : r)
	{
		data.insert(s);
	}

	EXPECT_EQ(data.size(), dims[0] * dims[1] * dims[2]);

	data.clear();

	auto extents = mesh.GetExtents();
	auto xmin = extents.first + (extents.second - extents.first) * 0.25;
	auto xmax = extents.first + (extents.second - extents.first) * 0.75;

	r = mesh.Select(VERTEX, xmin, xmax);

	for (auto s : r)
	{
		auto x = mesh.GetCoordinates(s);

		ASSERT_LE(xmin[0], x[0]);
		ASSERT_LE(xmin[1], x[1]);
		ASSERT_LE(xmin[2], x[2]);
		ASSERT_GE(xmax[0], x[0]);
		ASSERT_GE(xmax[1], x[1]);
		ASSERT_GE(xmax[2], x[2]);

		data.insert(s);
	}
	CHECK(data.size());

//	for (auto s : mesh.Select(VERTEX))
//	{
//		auto x = mesh.GetCoordinates(s);
//
//		if ((xmin[0] <= x[0]) && (xmin[1] <= x[1]) && (xmin[2] <= x[2]) && (xmax[0] >= x[0]) && (xmax[1] >= x[1])
//		        && (xmax[2] >= x[2]))
//		{
//			EXPECT_TRUE(data.find(s) != data.end()) << s;
//		}
//		else
//		{
//			EXPECT_TRUE(data.find(s) == data.end()) << s;
//		}
//
//	}

}

//
////TEST_P(TestMesh,scatter )
////{
////
////	Field<mesh_type, VERTEX, Real> n(mesh);
////	Field<mesh_type, EDGE, Real> J(mesh);
////
////	n.Clear();
////	J.Clear();
////
////	nTuple<3, Real> x = { -0.01, -0.01, -0.01 };
////	nTuple<3, Real> v = { 1, 2, 3 };
////
////	for (auto const & v : n)
////	{
////		std::cout << " " << v;
////	}
////	std::cout << std::endl;
////	for (auto const & v : J)
////	{
////		std::cout << " " << v;
////	}
////	std::cout << std::endl;
////	mesh.Scatter(x, 1.0, &n);
////	mesh.Scatter(x, v, &J);
////	for (auto const & v : n)
////	{
////		std::cout << " " << v;
////	}
////	std::cout << std::endl;
////	for (auto const & v : J)
////	{
////		std::cout << " " << v;
////	}
////
////	std::cout << std::endl;
////
////}
////
////TEST_P(TestMesh,gather)
////{
////
////}
//
////typedef testing::Types<RectMesh<>
//////, CoRectMesh<Complex>
////> AllMeshTypes;
////
////template<typename TMesh>
////class TestMesh: public testing::Test
////{
////protected:
////	virtual void SetUp()
////	{
////		Logger::Verbose(10);
////	}
////public:
////
////	typedef TMesh mesh_type;
////
////	DEFINE_FIELDS(mesh_type)
////};
////
////TYPED_TEST_CASE(TestMesh, AllMeshTypes);
////
////TYPED_TEST(TestMesh,create_default){
////{
////
////
////	mesh_type mesh;
////
////	mesh.Update();
////
////	LOGGER<<mesh;
////
////}
////}
////TYPED_TEST(TestMesh,Create_parse_cfg){
////{
////	LuaObject cfg;
////	cfg.ParseString(
////
////			" Grid=                                                                       \n"
////			" {                                                                                            \n"
////			"   Type=\"CoRectMesh\",                                                                       \n"
////			"   ScalarType=\"Real\",                                                                  \n"
////			"   UnitSystem={Type=\"SI\"},                                                                  \n"
////			"   Topology=                                                                                  \n"
////			"   {                                                                                          \n"
////			"       Type=\"3DCoRectMesh\",                                                                 \n"
////			"       Dimensions={100,100,100}, -- number of grid, now only first dimension is valid            \n"
////			"       GhostWidth= {5,0,0},  -- width of ghost points  , if gw=0, coordinate is               \n"
////			"                               -- Periodic at this direction                                  \n"
////			"   },                                                                                         \n"
////			"   Geometry=                                                                                  \n"
////			"   {                                                                                          \n"
////			"       Type=\"Origin_DxDyDz\",                                                                \n"
////			"       Min={0.0,0.0,0.0},                                                                     \n"
////			"       Max={1.0,1.0,1.0},                                                                     \n"
////			"       dt=0.5*1.0/ (100.0-1.0)  -- time step                                                      \n"
////			"   }                                                                                          \n"
////			"}                                                                                             \n"
////
////	);
////
////
////
////	mesh_type mesh;
////
////	mesh.Deserialize( cfg["Grid"]);
////
////	LOGGER<<mesh;
////
////}
////}
////
////template<typename TMesh>
////class TestMeshFunctions: public testing::Test
////{
////protected:
////	virtual void SetUp()
////	{
////		Logger::Verbose(10);
////
////		mesh.SetDt(1.0);
////
////		nTuple<3, Real> xmin = { 0, 0, 0 };
////		nTuple<3, Real> xmax = { 1, 1, 1 };
////		mesh.SetExtent(xmin, xmax);
////
////		nTuple<3, size_t> dims = { 20, 20, 20 };
////		mesh.SetDimensions(dims);
////
////		mesh.Update();
////
////	}
////public:
////
////	typedef TMesh mesh_type;
////
////	DEFINE_FIELDS(mesh_type)
////
////	mesh_type mesh;
////
////};
////
////TYPED_TEST_CASE(TestMeshFunctions, AllMeshTypes);
////
////TYPED_TEST(TestMeshFunctions,add_tags){
////{
////
////
////	mesh_type & mesh=mesh;
////
////	LuaObject cfg;
////
////	cfg.ParseString(
////			" Media=                                                                 "
////			" {                                                                      "
////			"    {Type=\"Vacuum\",Range={{0.2,0,0},{0.8,0,0}},Op=\"Set\"},          "
////			"                                                                        "
////			"    {Type=\"Plasma\",                                                   "
////			"      Select=function(x,y,z)                                            "
////			"           return x>1.0 and x<2.0 ;                                     "
////			"         end                                                            "
////			"     ,Op=\"Set\"},                                                      "
////			" }                                                                      "
////	);
////
////	mesh.tags().Deserialize(cfg["Media"]);
////
////}}
//
////*******************************************************************************************************
//
////template<typename TMesh>
////class Test1DMesh: public testing::Test
////{
////protected:
////	virtual void SetUp()
////	{
////		Logger::Verbose(10);
////
////		mesh.dt_ = 1.0;
////		mesh.xmin_[0] = 0;
////		mesh.xmin_[1] = 0;
////		mesh.xmin_[2] = 0;
////		mesh.xmax_[0] = 1.0;
////		mesh.xmax_[1] = 1.0;
////		mesh.xmax_[2] = 1.0;
////		mesh.dims_[0] = 20;
////		mesh.dims_[1] = 1;
////		mesh.dims_[2] = 1;
////		mesh.dt_ = 1.0;
////
////		mesh.Update();
////
////		GLOBAL_DATA_STREAM.OpenFile("");
////
////	}
////public:
////
////	typedef TMesh mesh_type;
////
////	DEFINE_FIELDS(mesh_type)
////
////	mesh_type mesh;
////
////};
////TYPED_TEST_CASE(Test1DMesh, AllMeshTypes);
////
////TYPED_TEST(Test1DMesh,shift){
////{
////
////
////	mesh_type & mesh=mesh;
////
////	EXPECT_EQ(mesh.Shift(mesh.INC(0),105),6L);
////	EXPECT_EQ(mesh.Shift(mesh.INC(1),105),5L);
////	EXPECT_EQ(mesh.Shift(mesh.INC(2),105),5L);
////
////	EXPECT_EQ(mesh.Shift(mesh.DES(0),105),4L);
////	EXPECT_EQ(mesh.Shift(mesh.DES(1),105),5L);
////	EXPECT_EQ(mesh.Shift(mesh.DES(2),105),5L);
////
////	auto s= mesh.GetIndex(3,4,5);
////	EXPECT_EQ(mesh.Shift(mesh.DES(0),3,4,5),s-1);
////	EXPECT_EQ(mesh.Shift(mesh.DES(1),3,4,5),s);
////	EXPECT_EQ(mesh.Shift(mesh.DES(2),3,4,5),s);
////
////	EXPECT_EQ(mesh.Shift(mesh.INC(0),3,4,5),s+1);
////	EXPECT_EQ(mesh.Shift(mesh.INC(1),3,4,5),s);
////	EXPECT_EQ(mesh.Shift(mesh.INC(2),3,4,5),s);
////
////}}

#endif /* MESH_TEST_H_ */
