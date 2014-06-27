/*
 * geometry_test.h
 *
 *  Created on: 2014年6月27日
 *      Author: salmon
 */

#ifndef GEOMETRY_TEST_H_
#define GEOMETRY_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../parallel/message_comm.h"

using namespace simpla;

#ifndef GEOMETRY
#include "../mesh/uniform_array.h"
#include "../mesh/geometry_cartesian.h"

typedef CartesianGeometry<UniformArray, false> TGeometry;
#else
typedef GEOMETRY TGeometry;
#endif

class TestGeometry: public testing::TestWithParam<
        std::tuple<typename TGeometry::coordinates_type, typename TGeometry::coordinates_type,
                nTuple<TGeometry::NDIMS, size_t> > >
{
protected:
	void SetUp()
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

		geometry.SetExtents(xmin,xmax,dims);

	}
public:
	typedef TGeometry geometry_type;
	typedef typename geometry_type::index_type index_type;
	typedef typename geometry_type::range_type range_type;
	typedef typename geometry_type::iterator iterator;
	typedef typename geometry_type::coordinates_type coordinates_type;

	unsigned int NDIMS=geometry_type::NDIMS;

	geometry_type geometry;

	std::vector<unsigned int> iform_list =
	{	VERTEX, EDGE, FACE, VOLUME};
	coordinates_type xmin,xmax;
	nTuple<geometry_type::NDIMS, index_type> dims;

};

//TEST_P(TestGeometry,misc)
//{
//	for (auto s : geometry.Select(VERTEX))
//	{
//		CHECK(geometry.GetCoordinates(s)) << dims;
//	}
//}

TEST_P(TestGeometry, coordinates)
{

//
//
//	auto range0 = geometry.Select(VERTEX);
//	auto range1 = geometry.Select(EDGE);
//	auto range2 = geometry.Select(FACE);
//	auto range3 = geometry.Select(VOLUME);
//
//	auto half_dx = geometry.GetDx() / 2;
//
//	EXPECT_EQ(extents.first, geometry.GetCoordinates(*begin(range0)));
//	EXPECT_EQ(extents.first + coordinates_type( { half_dx[0], 0, 0 }), geometry.GetCoordinates(*begin(range1)));
//	EXPECT_EQ(extents.first + coordinates_type( { 0, half_dx[1], half_dx[2] }),
//	        geometry.GetCoordinates(*begin(range2)));
//	EXPECT_EQ(extents.first + half_dx, geometry.GetCoordinates(*begin(range3)));

	auto extents = geometry.GetExtents();
	coordinates_type x = 0.21235 * (std::get<1>(extents) - std::get<0>(extents)) + std::get<0>(extents);

	for (auto iform : iform_list)
	{
		auto idx = geometry.CoordinatesGlobalToLocal(x, geometry.get_first_node_shift(iform));
		EXPECT_EQ(x, geometry.CoordinatesLocalToGlobal(idx) ) << "IForm =" << iform;

		auto s = std::get<0>(idx);
		EXPECT_EQ(iform, geometry.IForm(s));
		EXPECT_EQ(geometry.NodeId(geometry.get_first_node_shift(iform)), geometry.NodeId(s));
		EXPECT_EQ(geometry.ComponentNum(geometry.get_first_node_shift(iform)), geometry.ComponentNum(s));

		EXPECT_GE(3, InnerProductNTuple(std::get<1>(idx), std::get<1>(idx)));

	}

	auto idx = geometry.topology_type::CoordinatesToIndex(x);

	EXPECT_EQ(idx, geometry.topology_type::CoordinatesToIndex(geometry.topology_type::IndexToCoordinates(idx)));

}

TEST_P(TestGeometry, volume)
{
//	auto extents = geometry.GetExtents();
//	coordinates_type x = 0.21235 * (std::get<1>(extents) - std::get<0>(extents)) + std::get<0>(extents);
//
//	auto s0 = std::get<0>(geometry.CoordinatesGlobalToLocal(x, geometry.get_first_node_shift(VERTEX)));
//	auto s1 = std::get<0>(geometry.CoordinatesGlobalToLocal(x, geometry.get_first_node_shift(EDGE)));
//	auto s2 = std::get<0>(geometry.CoordinatesGlobalToLocal(x, geometry.get_first_node_shift(FACE)));
//	auto s3 = std::get<0>(geometry.CoordinatesGlobalToLocal(x, geometry.get_first_node_shift(VOLUME)));
//
////	CHECK(mesh.length_);
////	CHECK(mesh.Volume(s0));
////	CHECK(mesh.Volume(s1));
////	CHECK(mesh.NodeId(s2));
////
//	CHECK(geometry.volume_[0]);
//	CHECK(geometry.volume_[1]);
//	CHECK(geometry.volume_[2]);
//	CHECK(geometry.volume_[4]);
////	CHECK(mesh.volume_[mesh.NodeId(s2)]);
////	CHECK(mesh.GetCoordinates(s2));
////	CHECK(mesh.Volume(s2));
//
//	CHECK(geometry.topology_type::Volume(s0));
//	CHECK(geometry.topology_type::Volume(s1));
//	CHECK(geometry.topology_type::Volume(s2));
//	CHECK(geometry.topology_type::Volume(s3));
//
//	CHECK(geometry.GetCoordinates(s0));
//
//	CHECK(geometry.Volume(s0));
//	CHECK(geometry.Volume(s1));
//	CHECK(geometry.Volume(s2));
//	CHECK(geometry.Volume(s3));
//
//	EXPECT_DOUBLE_EQ(geometry.Volume(s0), geometry.DualVolume(s3));
//	EXPECT_DOUBLE_EQ(geometry.Volume(s1), geometry.DualVolume(s2));
//	EXPECT_DOUBLE_EQ(geometry.Volume(s0) * geometry.Volume(s3), geometry.Volume(s1) * geometry.Volume(s2));
//
//	auto d = geometry.GetDimensions();
//	EXPECT_DOUBLE_EQ(1.0, geometry.Volume(s0));
//	EXPECT_DOUBLE_EQ(d[0] <= 1 ? 1.0 : geometry.GetDx(s1)[0], geometry.Volume(s1)) << geometry.GetDx(s1)[2];
//	EXPECT_DOUBLE_EQ(d[1] <= 1 ? 1.0 : geometry.GetDx(geometry.Roate(s1))[1], geometry.Volume(geometry.Roate(s1)));
//	EXPECT_DOUBLE_EQ(d[2] <= 1 ? 1.0 : geometry.GetDx(geometry.InverseRoate(s1))[2],
//	        geometry.Volume(geometry.InverseRoate(s1)));

	for (auto iform : iform_list)
	{
		for (auto s : geometry.Select(iform))
		{
			auto IX = geometry_type::DI(0, s);
			auto IY = geometry_type::DI(1, s);
			auto IZ = geometry_type::DI(2, s);

			EXPECT_DOUBLE_EQ(geometry.CellVolume(s), geometry.DualVolume(s) * geometry.Volume(s));
			EXPECT_DOUBLE_EQ(1.0 / geometry.CellVolume(s), geometry.InvDualVolume(s) * geometry.InvVolume(s));

			EXPECT_DOUBLE_EQ(1.0, geometry.InvVolume(s) * geometry.Volume(s));
			EXPECT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s) * geometry.DualVolume(s));

			EXPECT_DOUBLE_EQ(1.0, geometry.InvVolume(s + IX) * geometry.Volume(s + IX));
			EXPECT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s + IX) * geometry.DualVolume(s + IX));

			EXPECT_DOUBLE_EQ(1.0, geometry.InvVolume(s - IY) * geometry.Volume(s - IY));
			EXPECT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s - IY) * geometry.DualVolume(s - IY));

			EXPECT_DOUBLE_EQ(1.0, geometry.InvVolume(s - IZ) * geometry.Volume(s - IZ));
			EXPECT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s - IZ) * geometry.DualVolume(s - IZ));

		}
	}

}

#endif /* GEOMETRY_TEST_H_ */
