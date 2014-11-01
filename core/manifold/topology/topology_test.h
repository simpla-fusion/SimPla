/*
 * topology_test.h
 *
 *  created on: 2014-6-27
 *      Author: salmon
 */

#ifndef TOPOLOGY_TEST_H_
#define TOPOLOGY_TEST_H_

#include <gtest/gtest.h>

#include "../../utilities/pretty_stream.h"
#include "../../utilities/log.h"

using namespace simpla;

//#ifndef TOPOLOGY
//#include "structured.h"
//typedef SurturedMesh TopologyType;
//#else
typedef TOPOLOGY TopologyType;
//#endif

class TestTopology: public testing::TestWithParam<
		nTuple<size_t, TopologyType::ndims> >
{
protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(10);

		dims = GetParam();

		topology.dimensions(dims);

	}
public:
	typedef TopologyType topology_type;
	typedef typename topology_type::index_type index_type;
	typedef typename topology_type::compact_index_type compact_index_type;
	typedef typename topology_type::iterator iterator;
	typedef typename topology_type::range_type range_type;
	typedef typename topology_type::coordinates_type coordinates_type;

	size_t NDIMS = topology_type::ndims;

	topology_type topology;

	std::vector<size_t> iform_list = { VERTEX, EDGE, FACE, VOLUME };

	nTuple<size_t, TopologyType::ndims> dims;

};

//TEST_P(TestTopology, ttest)
//{
//	coordinates_type x0 = { 0, 0, 0 }, x1 = { 1, 1, 1 };
//
//	CHECK(mesh.coordinates_to_index(x0));
//	CHECK(mesh.coordinates_to_index(x1));
//	auto d = mesh.delta_index(mesh.get_first_node_shift(EDGE));
//	auto s0 = mesh.compact(mesh.coordinates_to_index(x0)) | mesh.get_first_node_shift(VERTEX);
//	auto s1 = mesh.compact(mesh.coordinates_to_index(x1)) | mesh.get_first_node_shift(VERTEX);
//	CHECK_BIT(s0);
//	CHECK_BIT(d);
//	CHECK_BIT(s0 - d);
//	CHECK_BIT(s0 - d);
//	CHECK_BIT(s0 + d);
//	CHECK(mesh.Hash(s0));
//	CHECK(mesh.Hash(s0 - d));
//	CHECK(mesh.Hash(s0 + d));
//	CHECK(mesh.get_coordinates(s0 - d));
//	CHECK(mesh.get_coordinates(s0 + d));
//}

//TEST_P(TestTopology,misc)
//{
//	EXPECT_EQ(NProduct(dims), topology.get_num_of_elements());
//
//	for (auto s : topology.select(VERTEX))
//	{
//		CHECK(topology.get_coordinates(s)) << dims;
//	}
//}

TEST_P(TestTopology, compact_index_type)
{

	for (int depth = 0; depth < (topology_type::MAX_DEPTH_OF_TREE); ++depth)
	{
		for (int noid = 0; noid < 8; ++noid)
			ASSERT_EQ(noid, topology.node_id(topology.get_shift(noid, depth)));
	}

	auto s = topology.get_first_node_shift(VERTEX);
	EXPECT_EQ(0, topology.node_id(s));
	EXPECT_EQ(0, topology.node_id(topology.roate(s)));
	EXPECT_EQ(0, topology.node_id(topology.inverse_roate(s)));
	EXPECT_EQ(0, topology.component_number(topology.roate(s)));
	EXPECT_EQ(0, topology.component_number(topology.inverse_roate(s)));
	EXPECT_EQ(VERTEX, topology.node_id(s));
	EXPECT_EQ(VERTEX, topology.IForm(topology.roate(s)));
	EXPECT_EQ(VERTEX, topology.IForm(topology.inverse_roate(s)));

	s = topology.get_first_node_shift(VOLUME);

	EXPECT_EQ(7, topology.node_id(s));
	EXPECT_EQ(7, topology.node_id(topology.roate(s)));
	EXPECT_EQ(7, topology.node_id(topology.inverse_roate(s)));
	EXPECT_EQ(0, topology.component_number(topology.roate(s)));
	EXPECT_EQ(0, topology.component_number(topology.inverse_roate(s)));

	EXPECT_EQ(VOLUME, topology.IForm(s));
	EXPECT_EQ(VOLUME, topology.IForm(topology.roate(s)));
	EXPECT_EQ(VOLUME, topology.IForm(topology.inverse_roate(s)));

	s = topology.get_first_node_shift(EDGE);
	EXPECT_EQ(4, topology.node_id(s));
	EXPECT_EQ(2, topology.node_id(topology.roate(s)));
	EXPECT_EQ(1, topology.node_id(topology.roate(topology.roate(s))));
	EXPECT_EQ(1, topology.node_id(topology.inverse_roate(s)));
	EXPECT_EQ(2,
			topology.node_id(
					topology.inverse_roate(topology.inverse_roate(s))));

	EXPECT_EQ(0, topology.component_number(s));
	EXPECT_EQ(1, topology.component_number(topology.roate(s)));
	EXPECT_EQ(2, topology.component_number(topology.roate(topology.roate(s))));
	EXPECT_EQ(2, topology.component_number(topology.inverse_roate(s)));
	EXPECT_EQ(1,
			topology.component_number(
					topology.inverse_roate(topology.inverse_roate(s))));

	EXPECT_EQ(EDGE, topology.IForm(s));
	EXPECT_EQ(EDGE, topology.IForm(topology.roate(s)));
	EXPECT_EQ(EDGE, topology.IForm(topology.roate(topology.roate(s))));
	EXPECT_EQ(EDGE, topology.IForm(topology.inverse_roate(s)));
	EXPECT_EQ(EDGE,
			topology.IForm(topology.inverse_roate(topology.inverse_roate(s))));

	EXPECT_EQ(3, topology.node_id(topology.dual(s)));
	EXPECT_EQ(5, topology.node_id(topology.dual(topology.roate(s))));
	EXPECT_EQ(6, topology.node_id(topology.dual(topology.inverse_roate(s))));

	EXPECT_EQ(topology.DI(0, s), topology.delta_index(s));
	EXPECT_EQ(topology.DI(1, s), topology.delta_index(topology.roate(s)));
	EXPECT_EQ(topology.DI(2, s),
			topology.delta_index(topology.inverse_roate(s)));

	s = topology.get_first_node_shift(FACE);
	EXPECT_EQ(3, topology.node_id(s));
	EXPECT_EQ(5, topology.node_id(topology.roate(s)));
	EXPECT_EQ(6, topology.node_id(topology.roate(topology.roate(s))));
	EXPECT_EQ(6, topology.node_id(topology.inverse_roate(s)));
	EXPECT_EQ(5,
			topology.node_id(
					topology.inverse_roate(topology.inverse_roate(s))));

	EXPECT_EQ(0, topology.component_number(s));
	EXPECT_EQ(1, topology.component_number(topology.roate(s)));
	EXPECT_EQ(2, topology.component_number(topology.roate(topology.roate(s))));
	EXPECT_EQ(2, topology.component_number(topology.inverse_roate(s)));
	EXPECT_EQ(1,
			topology.component_number(
					topology.inverse_roate(topology.inverse_roate(s))));

	EXPECT_EQ(FACE, topology.IForm(s));
	EXPECT_EQ(FACE, topology.IForm(topology.roate(s)));
	EXPECT_EQ(FACE, topology.IForm(topology.roate(topology.roate(s))));
	EXPECT_EQ(FACE, topology.IForm(topology.inverse_roate(s)));
	EXPECT_EQ(FACE,
			topology.IForm(topology.inverse_roate(topology.inverse_roate(s))));

	EXPECT_EQ(4, topology.node_id(topology.dual(s)));
	EXPECT_EQ(2, topology.node_id(topology.dual(topology.roate(s))));
	EXPECT_EQ(1, topology.node_id(topology.dual(topology.inverse_roate(s))));

}

TEST_P(TestTopology, coordinates)
{

	auto extents = topology.extents();
	auto xmin = std::get<0>(extents);
	auto xmax = std::get<1>(extents);

	auto range0 = topology.select(VERTEX);
	auto range1 = topology.select(EDGE);
	auto range2 = topology.select(FACE);
	auto range3 = topology.select(VOLUME);

	auto half_dx = topology.dx() / 2;

	EXPECT_EQ(xmin, topology.coordinates(*begin(range0)));
	EXPECT_EQ(xmin + coordinates_type( { half_dx[0], 0, 0 }),
			topology.coordinates(*begin(range1)));
	EXPECT_EQ(xmin + coordinates_type( { 0, half_dx[1], half_dx[2] }),
			topology.coordinates(*begin(range2)));
	EXPECT_EQ(xmin + half_dx, topology.coordinates(*begin(range3)));

	typename topology_type::coordinates_type x = 0.21235 * (xmax - xmin) + xmin;
//
	for (auto iform : iform_list)

	{
		auto shift = topology.get_first_node_shift(iform);
		auto idx = topology.coordinates_global_to_local(x, shift);

		auto actual = topology.coordinates_local_to_global(idx);

		Real error = 0.0;

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] > 1)
				error += abs(x[i] - actual[i]);
		}
		error /= NDIMS;

		EXPECT_GT(EPSILON*1000 , error) << "IForm =" << iform << " " << x
				<< " ~~ " << actual << "  " << error;

		auto s = std::get<0>(idx);

		EXPECT_EQ(iform, topology.IForm(s));
		EXPECT_EQ(topology.node_id(shift), topology.node_id(s));
		EXPECT_EQ(topology.component_number(shift),
				topology.component_number(s));

		EXPECT_LT( dot(std::get<1>(idx), std::get<1>(idx)),3)
				<< std::get<1>(idx) << "IForm =" << iform;

	}
	auto idx = topology.coordinates_to_index(x);

	EXPECT_EQ(idx,
			topology.coordinates_to_index(topology.index_to_coordinates(idx)));

}

TEST_P(TestTopology, iterator)
{

	for (auto const & iform : iform_list)
	{
		EXPECT_EQ(topology.node_id(topology.get_first_node_shift(iform)),
				topology.node_id(*begin(topology.select(iform))));

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

//		auto hash = topology.make_hash(topology.select(iform));

		for (auto s : topology.select(iform))
		{
			++count;
			data.insert(topology.hash(s));
		}

		EXPECT_EQ(expect_count, data.size()) << iform;
		EXPECT_EQ(expect_count, count);
		EXPECT_EQ(0, *data.begin());
		EXPECT_EQ(expect_count - 1, *data.rbegin());

	}

}
TEST_P(TestTopology, hash)
{

	for (auto iform : iform_list)
	{
		size_t num = topology.get_local_memory_size(iform);

		auto domain = topology.select(iform);
		for (auto s : domain)
		{

			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s)),
					domain.hash(s));
			ASSERT_LE(0, domain.hash(s));

			ASSERT_GT(
					topology.get_local_memory_size(
							topology.IForm(topology.roate(s))),
					domain.hash(topology.roate(s)));
			ASSERT_LE(0, domain.hash(topology.inverse_roate(s)));

			auto DX = topology.DI(0, s);
			auto DY = topology.DI(1, s);
			auto DZ = topology.DI(2, s);

			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s + DX)),
					domain.hash(s + DX));
			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s + DY)),
					domain.hash(s + DY));
			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s + DZ)),
					domain.hash(s + DZ));
			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s - DX)),
					domain.hash(s - DX));
			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s - DY)),
					domain.hash(s - DY));
			ASSERT_GT(topology.get_local_memory_size(topology.IForm(s - DZ)),
					domain.hash(s - DZ));

			ASSERT_LE(0, domain.hash(s + DX));
			ASSERT_LE(0, domain.hash(s + DY));
			ASSERT_LE(0, domain.hash(s + DZ));
			ASSERT_LE(0, domain.hash(s - DX));
			ASSERT_LE(0, domain.hash(s - DY));
			ASSERT_LE(0, domain.hash(s - DZ));
		}

	}
}

TEST_P(TestTopology, split)
{

//	for (auto const & iform : iform_list)

	size_t iform = VERTEX;
	{

		nTuple<size_t, 3> begin = { 0, 0, 0 };

		nTuple<size_t, 3> end = dims;

		auto r = topology.make_range(begin, end,
				topology.get_first_node_shift(iform));

		size_t total = 4;

		std::set<compact_index_type> data;

		auto t = split(r, total, 1);

		for (int sub = 0; sub < total; ++sub)
			for (auto const & a : split(r, total, sub))
			{
//				nTuple<size_t, 3> idx = topology.decompact(a) >> 5;

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
#endif /* TOPOLOGY_TEST_H_ */
