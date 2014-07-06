/*
 * topology_test.h
 *
 *  Created on: 2014-6-27
 *      Author: salmon
 */

#ifndef TOPOLOGY_TEST_H_
#define TOPOLOGY_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"

using namespace simpla;

#ifndef TOPOLOGY
#include "../mesh/uniform_array.h"
typedef UniformArray TopologyType;
#else
typedef TOPOLOGY TopologyType;
#endif

class TestTopology: public testing::TestWithParam<std::tuple<nTuple<TopologyType::NDIMS, size_t> > >
{
protected:
	void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

		dims=std::get<0>(GetParam());

		topology.SetDimensions(dims);

	}
public:
	typedef TopologyType topology_type;
	typedef typename topology_type::index_type index_type;
	typedef typename topology_type::compact_index_type compact_index_type;
	typedef typename topology_type::iterator iterator;
	typedef typename topology_type::range_type range_type;
	typedef typename topology_type::coordinates_type coordinates_type;

	unsigned int NDIMS=topology_type::NDIMS;

	topology_type topology;

	std::vector<unsigned int> iform_list =
	{	VERTEX, EDGE, FACE, VOLUME};

	nTuple<TopologyType::NDIMS, index_type> dims;

};

//TEST_P(TestTopology, ttest)
//{
//	coordinates_type x0 = { 0, 0, 0 }, x1 = { 1, 1, 1 };
//
//	CHECK(mesh.CoordinatesToIndex(x0));
//	CHECK(mesh.CoordinatesToIndex(x1));
//	auto d = mesh.DeltaIndex(mesh.get_first_node_shift(EDGE));
//	auto s0 = mesh.Compact(mesh.CoordinatesToIndex(x0)) | mesh.get_first_node_shift(VERTEX);
//	auto s1 = mesh.Compact(mesh.CoordinatesToIndex(x1)) | mesh.get_first_node_shift(VERTEX);
//	CHECK_BIT(s0);
//	CHECK_BIT(d);
//	CHECK_BIT(s0 - d);
//	CHECK_BIT(s0 - d);
//	CHECK_BIT(s0 + d);
//	CHECK(mesh.Hash(s0));
//	CHECK(mesh.Hash(s0 - d));
//	CHECK(mesh.Hash(s0 + d));
//	CHECK(mesh.GetCoordinates(s0 - d));
//	CHECK(mesh.GetCoordinates(s0 + d));
//}

//TEST_P(TestTopology,misc)
//{
//	EXPECT_EQ(NProduct(dims), topology.GetNumOfElements());
//
//	for (auto s : topology.Select(VERTEX))
//	{
//		CHECK(topology.GetCoordinates(s)) << dims;
//	}
//}

TEST_P(TestTopology, compact_index_type)
{

	for (int depth = 0; depth < topology_type::MAX_DEPTH_OF_TREE; ++depth)
	{
		for (int noid = 0; noid < 8; ++noid)
			ASSERT_EQ(noid, topology.NodeId(topology.GetShift(noid, depth)));
	}

	auto s = topology.get_first_node_shift(VERTEX);
	EXPECT_EQ(0, topology.NodeId(s));
	EXPECT_EQ(0, topology.NodeId(topology.Roate(s)));
	EXPECT_EQ(0, topology.NodeId(topology.InverseRoate(s)));
	EXPECT_EQ(0, topology.ComponentNum(topology.Roate(s)));
	EXPECT_EQ(0, topology.ComponentNum(topology.InverseRoate(s)));
	EXPECT_EQ(VERTEX, topology.NodeId(s));
	EXPECT_EQ(VERTEX, topology.IForm(topology.Roate(s)));
	EXPECT_EQ(VERTEX, topology.IForm(topology.InverseRoate(s)));

	s = topology.get_first_node_shift(VOLUME);

	EXPECT_EQ(7, topology.NodeId(s));
	EXPECT_EQ(7, topology.NodeId(topology.Roate(s)));
	EXPECT_EQ(7, topology.NodeId(topology.InverseRoate(s)));
	EXPECT_EQ(0, topology.ComponentNum(topology.Roate(s)));
	EXPECT_EQ(0, topology.ComponentNum(topology.InverseRoate(s)));

	EXPECT_EQ(VOLUME, topology.IForm(s));
	EXPECT_EQ(VOLUME, topology.IForm(topology.Roate(s)));
	EXPECT_EQ(VOLUME, topology.IForm(topology.InverseRoate(s)));

	s = topology.get_first_node_shift(EDGE);
	EXPECT_EQ(4, topology.NodeId(s));
	EXPECT_EQ(2, topology.NodeId(topology.Roate(s)));
	EXPECT_EQ(1, topology.NodeId(topology.Roate(topology.Roate(s))));
	EXPECT_EQ(1, topology.NodeId(topology.InverseRoate(s)));
	EXPECT_EQ(2, topology.NodeId(topology.InverseRoate(topology.InverseRoate(s))));

	EXPECT_EQ(0, topology.ComponentNum(s));
	EXPECT_EQ(1, topology.ComponentNum(topology.Roate(s)));
	EXPECT_EQ(2, topology.ComponentNum(topology.Roate(topology.Roate(s))));
	EXPECT_EQ(2, topology.ComponentNum(topology.InverseRoate(s)));
	EXPECT_EQ(1, topology.ComponentNum(topology.InverseRoate(topology.InverseRoate(s))));

	EXPECT_EQ(EDGE, topology.IForm(s));
	EXPECT_EQ(EDGE, topology.IForm(topology.Roate(s)));
	EXPECT_EQ(EDGE, topology.IForm(topology.Roate(topology.Roate(s))));
	EXPECT_EQ(EDGE, topology.IForm(topology.InverseRoate(s)));
	EXPECT_EQ(EDGE, topology.IForm(topology.InverseRoate(topology.InverseRoate(s))));

	EXPECT_EQ(3, topology.NodeId(topology.Dual(s)));
	EXPECT_EQ(5, topology.NodeId(topology.Dual(topology.Roate(s))));
	EXPECT_EQ(6, topology.NodeId(topology.Dual(topology.InverseRoate(s))));

	EXPECT_EQ(topology.DI(0, s), topology.DeltaIndex(s));
	EXPECT_EQ(topology.DI(1, s), topology.DeltaIndex(topology.Roate(s)));
	EXPECT_EQ(topology.DI(2, s), topology.DeltaIndex(topology.InverseRoate(s)));

	s = topology.get_first_node_shift(FACE);
	EXPECT_EQ(3, topology.NodeId(s));
	EXPECT_EQ(5, topology.NodeId(topology.Roate(s)));
	EXPECT_EQ(6, topology.NodeId(topology.Roate(topology.Roate(s))));
	EXPECT_EQ(6, topology.NodeId(topology.InverseRoate(s)));
	EXPECT_EQ(5, topology.NodeId(topology.InverseRoate(topology.InverseRoate(s))));

	EXPECT_EQ(0, topology.ComponentNum(s));
	EXPECT_EQ(1, topology.ComponentNum(topology.Roate(s)));
	EXPECT_EQ(2, topology.ComponentNum(topology.Roate(topology.Roate(s))));
	EXPECT_EQ(2, topology.ComponentNum(topology.InverseRoate(s)));
	EXPECT_EQ(1, topology.ComponentNum(topology.InverseRoate(topology.InverseRoate(s))));

	EXPECT_EQ(FACE, topology.IForm(s));
	EXPECT_EQ(FACE, topology.IForm(topology.Roate(s)));
	EXPECT_EQ(FACE, topology.IForm(topology.Roate(topology.Roate(s))));
	EXPECT_EQ(FACE, topology.IForm(topology.InverseRoate(s)));
	EXPECT_EQ(FACE, topology.IForm(topology.InverseRoate(topology.InverseRoate(s))));

	EXPECT_EQ(4, topology.NodeId(topology.Dual(s)));
	EXPECT_EQ(2, topology.NodeId(topology.Dual(topology.Roate(s))));
	EXPECT_EQ(1, topology.NodeId(topology.Dual(topology.InverseRoate(s))));

}
TEST_P(TestTopology, hash)
{

	for (auto iform : iform_list)
	{
		size_t num = topology.GetLocalMemorySize(iform);

		auto hash = topology.make_hash(topology.Select(iform));

		for (auto s : topology.Select(iform))
		{

			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s)), hash(s));
			ASSERT_LE(0, hash(s));

			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(topology.Roate(s))), hash(topology.Roate(s)));
			ASSERT_LE(0, hash(topology.InverseRoate(s)));

			auto DX = topology.DI(0, s);
			auto DY = topology.DI(1, s);
			auto DZ = topology.DI(2, s);

			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s + DX)), hash(s + DX));
			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s + DY)), hash(s + DY));
			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s + DZ)), hash(s + DZ));
			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s - DX)), hash(s - DX));
			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s - DY)), hash(s - DY));
			ASSERT_GT(topology.GetLocalMemorySize(topology.IForm(s - DZ)), hash(s - DZ));

			ASSERT_LE(0, hash(s + DX));
			ASSERT_LE(0, hash(s + DY));
			ASSERT_LE(0, hash(s + DZ));
			ASSERT_LE(0, hash(s - DX));
			ASSERT_LE(0, hash(s - DY));
			ASSERT_LE(0, hash(s - DZ));
		}

	}
}

TEST_P(TestTopology, iterator)
{

	for (auto const & iform : iform_list)
	{
		EXPECT_EQ(topology.NodeId(topology.get_first_node_shift(iform)),
		        topology.NodeId(*begin(topology.Select(iform))));

		size_t expect_count = 1;

		for (int i = 0; i < NDIMS; ++i)
		{
			expect_count *= dims[i];
		}
		if (iform == EDGE || iform == FACE)
		{
			expect_count *= 3;
		}

		std::set<compact_index_type> data;

		size_t count = 0;

		auto hash = topology.make_hash(topology.Select(iform));

		for (auto s : topology.Select(iform))
		{
			++count;
			data.insert(hash(s));
		}

		EXPECT_EQ(expect_count, data.size()) << iform;
		EXPECT_EQ(expect_count, count);
		EXPECT_EQ(0, *data.begin());
		EXPECT_EQ(expect_count - 1, *data.rbegin());

	}

}

TEST_P(TestTopology, Split)
{

//	for (auto const & iform : iform_list)

	unsigned int iform = VERTEX;
	{

		nTuple<3, index_type> begin = { 0, 0, 0 };

		nTuple<3, index_type> end = dims;

		auto r = topology.make_range(begin, end, topology.get_first_node_shift(iform));

		size_t total = 4;

		std::set<compact_index_type> data;

		for (int sub = 0; sub < total; ++sub)
			for (auto const & a : Split(r, total, sub))
			{
				data.insert(a);
			}

		size_t size = NProduct(dims);

		if (iform == VERTEX || iform == VOLUME)
		{
			ASSERT_EQ(data.size(), size);
		}
		else
		{
			ASSERT_EQ(data.size(), size * 3);
		}
	}

}

TEST_P(TestTopology, coordinates)
{

	auto extents = topology.GetExtents();
	auto xmin = std::get<0>(extents);
	auto xmax = std::get<1>(extents);

	auto range0 = topology.Select(VERTEX);
	auto range1 = topology.Select(EDGE);
	auto range2 = topology.Select(FACE);
	auto range3 = topology.Select(VOLUME);

	auto half_dx = topology.GetDx() / 2;

	EXPECT_EQ(xmin, topology.GetCoordinates(*begin(range0)));
	EXPECT_EQ(xmin + coordinates_type( { half_dx[0], 0, 0 }), topology.GetCoordinates(*begin(range1)));
	EXPECT_EQ(xmin + coordinates_type( { 0, half_dx[1], half_dx[2] }), topology.GetCoordinates(*begin(range2)));
	EXPECT_EQ(xmin + half_dx, topology.GetCoordinates(*begin(range3)));

	typename topology_type::coordinates_type x = 0.21235 * (xmax - xmin) + xmin;

	for (auto iform : iform_list)
	{
		auto shift = topology.get_first_node_shift(iform);
		auto idx = topology.CoordinatesGlobalToLocal(x, shift);
		EXPECT_EQ(x, topology.CoordinatesLocalToGlobal(idx) ) << "IForm =" << iform;

		auto s = std::get<0>(idx);

		EXPECT_EQ(iform, topology.IForm(s));
		EXPECT_EQ(topology.NodeId(shift), topology.NodeId(s));
		EXPECT_EQ(topology.ComponentNum(shift), topology.ComponentNum(s));

		EXPECT_LT( InnerProductNTuple(std::get<1>(idx), std::get<1>(idx)),3) << std::get<1>(idx) << "IForm =" << iform;

	}

	auto idx = topology.CoordinatesToIndex(x);

	EXPECT_EQ(idx, topology.CoordinatesToIndex(topology.IndexToCoordinates(idx)));

}
#endif /* TOPOLOGY_TEST_H_ */
