/*
 * geometry_test.h
 *
 *  Created on: 2014-6-27
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

typedef CartesianGeometry<UniformArray> TGeometry;
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
		LOG_STREAM.set_stdout_visable_level(10);

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

		geometry.set_extents(xmin,xmax,dims);

	}
public:
	typedef TGeometry geometry_type;
	typedef typename geometry_type::index_type index_type;
	typedef typename geometry_type::range_type range_type;
	typedef typename geometry_type::scalar_type scalar_type;
	typedef typename geometry_type::iterator iterator;
	typedef typename geometry_type::coordinates_type coordinates_type;

	  unsigned int   NDIMS=geometry_type::NDIMS;

	geometry_type geometry;

	std::vector< unsigned int  > iform_list =
	{	VERTEX, EDGE, FACE, VOLUME};
	coordinates_type xmin,xmax;
	nTuple<geometry_type::NDIMS, index_type> dims;

	Real epsilon=EPSILON*10;

};

TEST_P(TestGeometry, Coordinates)
{

	auto extents = geometry.get_extents();
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

TEST_P(TestGeometry, Volume)
{
//	auto extents = geometry.get_extents();
//	coordinates_type x = 0.21235 * (std::get<1>(extents) - std::get<0>(extents)) + std::get<0>(extents);

	for (auto iform : iform_list)
	{
		for (auto s : geometry.Select(iform))
		{
			auto IX = geometry_type::DI(0, s);
			auto IY = geometry_type::DI(1, s);
			auto IZ = geometry_type::DI(2, s);

			ASSERT_DOUBLE_EQ(geometry.CellVolume(s), geometry.DualVolume(s) * geometry.Volume(s));
			ASSERT_DOUBLE_EQ(1.0 / geometry.CellVolume(s), geometry.InvDualVolume(s) * geometry.InvVolume(s));

			ASSERT_DOUBLE_EQ(1.0, geometry.InvVolume(s) * geometry.Volume(s));
			ASSERT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s) * geometry.DualVolume(s));

			ASSERT_DOUBLE_EQ(1.0, geometry.InvVolume(s + IX) * geometry.Volume(s + IX));
			ASSERT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s + IX) * geometry.DualVolume(s + IX));

			ASSERT_DOUBLE_EQ(1.0, geometry.InvVolume(s - IY) * geometry.Volume(s - IY));
			ASSERT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s - IY) * geometry.DualVolume(s - IY));

			ASSERT_DOUBLE_EQ(1.0, geometry.InvVolume(s - IZ) * geometry.Volume(s - IZ));
			ASSERT_DOUBLE_EQ(1.0, geometry.InvDualVolume(s - IZ) * geometry.DualVolume(s - IZ));
		}
	}

	auto extents = geometry.get_extents();
	coordinates_type x = 0.21235 * (std::get<1>(extents) - std::get<0>(extents)) + std::get<0>(extents);
	auto idx = geometry.CoordinatesGlobalToLocal(x, geometry.get_first_node_shift(VERTEX));

	auto s = std::get<0>(idx);
	auto IX = geometry_type::DI(0, s) << 1;
	auto IY = geometry_type::DI(1, s) << 1;
	auto IZ = geometry_type::DI(2, s) << 1;

//	CHECK_BIT(s);
//	CHECK_BIT(IX);
//	CHECK(geometry.Volume(s - IX));
//	CHECK(geometry.Volume(s + IX));
//	CHECK(geometry.Volume(s - IY));
//	CHECK(geometry.Volume(s + IY));
//	CHECK(geometry.Volume(s - IZ));
//	CHECK(geometry.Volume(s + IZ));
}

TEST_P(TestGeometry,Coordinates_transform)
{

	nTuple<3, Real> v = { 1.0, 2.0, 3.0 };
	auto extents = geometry.get_extents();
	coordinates_type x = 0.21235 * (std::get<1>(extents) - std::get<0>(extents)) + std::get<0>(extents);
	auto z = std::make_tuple(x, v);

	coordinates_type y = geometry.InvMapTo(geometry.MapTo(x));
	EXPECT_DOUBLE_EQ(x[0], y[0]);
	EXPECT_DOUBLE_EQ(x[1], y[1]);
	EXPECT_DOUBLE_EQ(x[2], y[2]);

	y = geometry.MapTo(geometry.InvMapTo(x));
	EXPECT_DOUBLE_EQ(x[0], y[0]);
	EXPECT_DOUBLE_EQ(x[1], y[1]);
	EXPECT_DOUBLE_EQ(x[2], y[2]);

	auto z1 = geometry.PushForward(geometry.PullBack(z));
	EXPECT_DOUBLE_EQ(std::get<0>(z)[0], std::get<0>(z1)[0]);
	EXPECT_DOUBLE_EQ(std::get<0>(z)[1], std::get<0>(z1)[1]);
	EXPECT_DOUBLE_EQ(std::get<0>(z)[2], std::get<0>(z1)[2]);
	EXPECT_DOUBLE_EQ(std::get<1>(z)[0], std::get<1>(z1)[0]);
	EXPECT_DOUBLE_EQ(std::get<1>(z)[1], std::get<1>(z1)[1]);
	EXPECT_DOUBLE_EQ(std::get<1>(z)[2], std::get<1>(z1)[2]);

	auto z2 = geometry.PullBack(geometry.PushForward(z));
	EXPECT_DOUBLE_EQ(std::get<0>(z)[0], std::get<0>(z2)[0]);
	EXPECT_DOUBLE_EQ(std::get<0>(z)[1], std::get<0>(z2)[1]);
	EXPECT_DOUBLE_EQ(std::get<0>(z)[2], std::get<0>(z2)[2]);
	EXPECT_DOUBLE_EQ(std::get<1>(z)[0], std::get<1>(z2)[0]);
	EXPECT_DOUBLE_EQ(std::get<1>(z)[1], std::get<1>(z2)[1]);
	EXPECT_DOUBLE_EQ(std::get<1>(z)[2], std::get<1>(z2)[2]);

}

#endif /* GEOMETRY_TEST_H_ */
