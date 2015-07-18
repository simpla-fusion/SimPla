/*
 * topology_test.h
 *
 *  created on: 2014-6-27
 *      Author: salmon
 */

#ifndef TOPOLOGY_TEST_H_
#define TOPOLOGY_TEST_H_

#include <gtest/gtest.h>

#include "utilities.h"
using namespace simpla;

typedef TOPOLOGY TopologyType;

TEST(TestTopologyStatic, id_type)
{
	typedef TopologyType t;

	for (int depth = 0; depth < (t::FLOATING_POINT_POS); ++depth)
	{
		for (int noid = 0; noid < 8; ++noid)
			ASSERT_EQ(noid, t::ele_suffix(t::get_shift(noid, depth)));
	}

	auto s = t::get_first_node_shift(VERTEX);
	EXPECT_EQ(0, t::ele_suffix(s));
	EXPECT_EQ(0, t::ele_suffix(t::rotate(s)));
	EXPECT_EQ(0, t::ele_suffix(t::inverse_rotate(s)));
	EXPECT_EQ(0, t::component_number(t::rotate(s)));
	EXPECT_EQ(0, t::component_number(t::inverse_rotate(s)));
	EXPECT_EQ(VERTEX, t::ele_suffix(s));
	EXPECT_EQ(VERTEX, t::IForm(t::rotate(s)));
	EXPECT_EQ(VERTEX, t::IForm(t::inverse_rotate(s)));

	s = t::get_first_node_shift(VOLUME);
	EXPECT_EQ(7, t::ele_suffix(s));
	EXPECT_EQ(7, t::ele_suffix(t::rotate(s)));
	EXPECT_EQ(7, t::ele_suffix(t::inverse_rotate(s)));
	EXPECT_EQ(0, t::component_number(t::rotate(s)));
	EXPECT_EQ(0, t::component_number(t::inverse_rotate(s)));

	EXPECT_EQ(VOLUME, t::IForm(s));
	EXPECT_EQ(VOLUME, t::IForm(t::rotate(s)));
	EXPECT_EQ(VOLUME, t::IForm(t::inverse_rotate(s)));

	s = t::get_first_node_shift(EDGE);
	EXPECT_EQ(1, t::ele_suffix(s));
	EXPECT_EQ(2, t::ele_suffix(t::rotate(s)));
	EXPECT_EQ(4, t::ele_suffix(t::rotate(t::rotate(s))));
	EXPECT_EQ(4, t::ele_suffix(t::inverse_rotate(s)));
	EXPECT_EQ(2, t::ele_suffix(t::inverse_rotate(t::inverse_rotate(s))));

	EXPECT_EQ(0, t::component_number(s));
	EXPECT_EQ(1, t::component_number(t::rotate(s)));
	EXPECT_EQ(2, t::component_number(t::rotate(t::rotate(s))));
	EXPECT_EQ(2, t::component_number(t::inverse_rotate(s)));
	EXPECT_EQ(1, t::component_number(t::inverse_rotate(t::inverse_rotate(s))));

	EXPECT_EQ(EDGE, t::IForm(s));
	EXPECT_EQ(EDGE, t::IForm(t::rotate(s)));
	EXPECT_EQ(EDGE, t::IForm(t::rotate(t::rotate(s))));
	EXPECT_EQ(EDGE, t::IForm(t::inverse_rotate(s)));
	EXPECT_EQ(EDGE, t::IForm(t::inverse_rotate(t::inverse_rotate(s))));

	EXPECT_EQ(6, t::ele_suffix(t::dual(s)));
	EXPECT_EQ(5, t::ele_suffix(t::dual(t::rotate(s))));
	EXPECT_EQ(3, t::ele_suffix(t::dual(t::inverse_rotate(s))));

	EXPECT_EQ(t::DI(0, s), t::delta_index(s));
	EXPECT_EQ(t::DI(1, s), t::delta_index(t::rotate(s)));
	EXPECT_EQ(t::DI(2, s), t::delta_index(t::inverse_rotate(s)));

	s = t::get_first_node_shift(FACE);
	EXPECT_EQ(6, t::ele_suffix(s));
	EXPECT_EQ(5, t::ele_suffix(t::rotate(s)));
	EXPECT_EQ(3, t::ele_suffix(t::rotate(t::rotate(s))));
	EXPECT_EQ(3, t::ele_suffix(t::inverse_rotate(s)));
	EXPECT_EQ(5, t::ele_suffix(t::inverse_rotate(t::inverse_rotate(s))));

	EXPECT_EQ(0, t::component_number(s));
	EXPECT_EQ(1, t::component_number(t::rotate(s)));
	EXPECT_EQ(2, t::component_number(t::rotate(t::rotate(s))));
	EXPECT_EQ(2, t::component_number(t::inverse_rotate(s)));
	EXPECT_EQ(1, t::component_number(t::inverse_rotate(t::inverse_rotate(s))));

	EXPECT_EQ(FACE, t::IForm(s));
	EXPECT_EQ(FACE, t::IForm(t::rotate(s)));
	EXPECT_EQ(FACE, t::IForm(t::rotate(t::rotate(s))));
	EXPECT_EQ(FACE, t::IForm(t::inverse_rotate(s)));
	EXPECT_EQ(FACE, t::IForm(t::inverse_rotate(t::inverse_rotate(s))));

	EXPECT_EQ(1, t::ele_suffix(t::dual(s)));
	EXPECT_EQ(2, t::ele_suffix(t::dual(t::rotate(s))));
	EXPECT_EQ(4, t::ele_suffix(t::dual(t::inverse_rotate(s))));

}
TEST(TestTopologyStatic, coordinates)
{
//	Topology.dimensions(&dims[0]);
//
//	auto extents = Topology.extents();
//	auto xmin = std::get<0>(extents);
//	auto xmax = std::get<1>(extents);
//
//	auto range0 = Topology.template select<VERTEX>();
//	auto range1 = Topology.template select<EDGE>();
//	auto range2 = Topology.template select<FACE>();
//	auto range3 = Topology.template select<VOLUME>();
//
//	auto half_dx = Topology.dx() / 2;
//
//	EXPECT_EQ(xmin, t::id_to_coordinates(*begin(range0)));
//	EXPECT_EQ(xmin + coordinate_tuple( { half_dx[0], 0, 0 }),
//			t::id_to_coordinates(*begin(range1)));
//	EXPECT_EQ(xmin + coordinate_tuple( { 0, half_dx[1], half_dx[2] }),
//			t::id_to_coordinates(*begin(range2)));
//	EXPECT_EQ(xmin + half_dx, t::id_to_coordinates(*begin(range3)));
	typedef TopologyType t;
	typename t::coordinate_tuple x =
	{ 0.21235, 1.2343, 0.1 };

	EXPECT_EQ(x,
			t::coordinates_local_to_global(
					t::coordinates_global_to_local(x, 0UL)));

	EXPECT_EQ(x,
			t::coordinates_local_to_global(
					t::coordinates_global_to_local(x, t::_DI)));

	EXPECT_EQ(x,
			t::coordinates_local_to_global(
					t::coordinates_global_to_local(x, t::_DI | t::_DK)));

	EXPECT_EQ(x,
			t::coordinates_local_to_global(
					t::coordinates_global_to_local(x, t::_DI)));

	EXPECT_EQ(x,
			t::coordinates_local_to_global(
					t::coordinates_global_to_local(x,
							t::_DI | t::_DJ | t::_DK)));

	auto idx = t::coordinates_to_index(x);

	EXPECT_EQ(idx, t::coordinates_to_index(t::index_to_coordinates(idx)));

	x = t::index_to_coordinates(idx);

	EXPECT_EQ(x, t::index_to_coordinates(t::coordinates_to_index(x)));

}
class TestTopology: public testing::TestWithParam<
		nTuple<size_t, TopologyType::ndims> >
{
protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(10);

		dims = GetParam();
	}
public:
	typedef TopologyType topology_type;
	typedef topology_type t;
	typedef typename topology_type::id_type id_type;
	typedef typename topology_type::index_type index_type;
	typedef typename topology_type::index_tuple index_tuple;

	typedef typename topology_type::coordinate_tuple coordinate_tuple;

	size_t NDIMS = topology_type::ndims;

	topology_type topology;

	index_tuple dims;

};

//TEST_P(TestTopology, ttest)
//{
//	coordinate_tuple x0 = { 0, 0, 0 }, x1 = { 1, 1, 1 };
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
//	EXPECT_EQ(NProduct(dims), t::get_num_of_elements());
//
//	for (auto s : t::select(VERTEX))
//	{
//		CHECK(t::get_coordinates(s)) << dims;
//	}
//}

TEST_P(TestTopology, foreach)
{
	topology.dimensions(&dims[0]);

	EXPECT_EQ(t::ele_suffix(t::get_first_node_shift(VERTEX)),
			t::ele_suffix(*begin(topology.template select<VERTEX>())));

	size_t expect_count = 1;

	for (int i = 0; i < NDIMS; ++i)
	{
		expect_count *= dims[i];
	}

	std::set<size_t> data;

	size_t count = 0;

	for (auto s : topology.template select<VERTEX>())
	{
		++count;

		data.insert(topology.hash(s));
	}

	EXPECT_EQ(expect_count, data.size()) << VERTEX;
	EXPECT_EQ(expect_count, count);
	EXPECT_EQ(0, *data.begin());
	EXPECT_EQ(expect_count - 1, *data.rbegin());

//#define BODY(iform)                                                                \
//{                                                                                  \
//	EXPECT_EQ(t::ele_suffix(t::get_first_node_shift(iform)),              \
//			t::ele_suffix(*begin(t::template select<iform>())));          \
//                                                                                   \
//	size_t expect_count = 1;                                                       \
//                                                                                   \
//	for (int i = 0; i < NDIMS; ++i)                                                \
//	{                                                                              \
//		expect_count *= dims[i];                                                   \
//	}                                                                              \
//	if (iform == EDGE || iform == FACE)                                            \
//	{                                                                              \
//		expect_count *= 3;                                                         \
//	}                                                                              \
//                                                                                   \
//	std::set<index_type> data;                                             \
//                                                                                   \
//	size_t count = 0;                                                              \
//                                                                                   \
//	for (auto s : t::template select<iform>())                               \
//	{                                                                              \
//		++count;                                                                   \
//		data.insert(t::hash(s));                                             \
//	}                                                                              \
//                                                                                   \
//	EXPECT_EQ(expect_count, data.size()) << iform;                                 \
//	EXPECT_EQ(expect_count, count);                                                \
//	EXPECT_EQ(0, *data.begin());                                                   \
//	EXPECT_EQ(expect_count - 1, *data.rbegin());                                   \
//}

//	BODY(VERTEX);
//	BODY(EDGE);
//	BODY(FACE);
//	BODY(VOLUME);

#undef BODY

}
//TEST_P(TestTopology, hash)
//{
////
////	for (auto iform : iform_list)
////	{
////		size_t num = t::get_local_memory_size(iform);
////
////		auto domain = t::select(iform);
////		for (auto s : domain)
////		{
////
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s)),
////					domain.hash(s));
////			ASSERT_LE(0, domain.hash(s));
////
////			ASSERT_GT(
////					t::get_local_memory_size(
////							t::IForm(t::rotate(s))),
////					domain.hash(t::rotate(s)));
////			ASSERT_LE(0, domain.hash(t::inverse_rotate(s)));
////
////			auto DX = t::DI(0, s);
////			auto DY = t::DI(1, s);
////			auto DZ = t::DI(2, s);
////
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s + DX)),
////					domain.hash(s + DX));
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s + DY)),
////					domain.hash(s + DY));
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s + DZ)),
////					domain.hash(s + DZ));
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s - DX)),
////					domain.hash(s - DX));
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s - DY)),
////					domain.hash(s - DY));
////			ASSERT_GT(t::get_local_memory_size(t::IForm(s - DZ)),
////					domain.hash(s - DZ));
////
////			ASSERT_LE(0, domain.hash(s + DX));
////			ASSERT_LE(0, domain.hash(s + DY));
////			ASSERT_LE(0, domain.hash(s + DZ));
////			ASSERT_LE(0, domain.hash(s - DX));
////			ASSERT_LE(0, domain.hash(s - DY));
////			ASSERT_LE(0, domain.hash(s - DZ));
////		}
////
////	}
//}
//
//TEST_P(TestTopology, split)
//{
//
////	for (auto const & iform : iform_list)
////
////	size_t iform = VERTEX;
////	{
////
////		index_tuple begin = { 0, 0, 0 };
////
////		index_tuple end = dims;
////
////		auto r = t::make_range(begin, end,
////				t::get_first_node_shift(iform));
////
////		size_t total = 4;
////
////		std::set<index_type> data;
////
////		auto t = split(r, total, 1);
////
////		for (int sub = 0; sub < total; ++sub)
////			for (auto const & a : split(r, total, sub))
////			{
//////				nTuple<size_t, 3> idx = t::decompact(a) >> 5;
////
////				data.insert(a);
////			}
////
////		size_t size = NProduct(dims);
////
////		if (iform == VERTEX || iform == VOLUME)
////		{
////			ASSERT_EQ(data.size(), size);
////		}
////		else
////		{
////			ASSERT_EQ(data.size(), size * 3);
////		}
////	}
//
//}
//
//TEST_P(TestTopology, volume)
//{
//
////	for (auto iform : iform_list)
////	{
////		for (auto s : t::select(iform))
////		{
////			auto IX = topology_type::DI(0, s);
////			auto IY = topology_type::DI(1, s);
////			auto IZ = topology_type::DI(2, s);
////
////			ASSERT_DOUBLE_EQ(t::cell_volume(s),
////					t::dual_volume(s) * t::volume(s));
////			ASSERT_DOUBLE_EQ(1.0 / t::cell_volume(s),
////					t::inv_dual_volume(s) * t::inv_volume(s));
////
////			ASSERT_DOUBLE_EQ(1.0, t::inv_volume(s) * t::volume(s));
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_dual_volume(s) * t::dual_volume(s));
////
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_volume(s + IX) * t::volume(s + IX));
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_dual_volume(s + IX)
////							* t::dual_volume(s + IX));
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_volume(s - IY) * t::volume(s - IY));
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_dual_volume(s - IY)
////							* t::dual_volume(s - IY));
////
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_volume(s - IZ) * t::volume(s - IZ));
////			ASSERT_DOUBLE_EQ(1.0,
////					t::inv_dual_volume(s - IZ)
////							* t::dual_volume(s - IZ));
////
////		}
////	}
//
////	auto extents = t::extents();
////	coordinate_tuple x = 0.21235 * (std::get<1>(extents) - std::get<0>(extents))
////			+ std::get<0>(extents);
////	auto idx = t::coordinates_global_to_local(x,
////			t::get_first_node_shift(VERTEX));
////
////	auto s = std::get<0>(idx);
////	auto IX = topology_type::DI(0, s) << 1;
////	auto IY = topology_type::DI(1, s) << 1;
////	auto IZ = topology_type::DI(2, s) << 1;
////
////	CHECK_BIT(s);
////	CHECK_BIT(IX);
////	CHECK(t::volume(s - IX));
////	CHECK(t::volume(s + IX));
////	CHECK(t::volume(s - IY));
////	CHECK(t::volume(s + IY));
////	CHECK(t::volume(s - IZ));
////	CHECK(t::volume(s + IZ));
//}
#endif /* TOPOLOGY_TEST_H_ */
