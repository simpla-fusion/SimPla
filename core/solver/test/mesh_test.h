/**
 * @file mesh_test.h
 *
 * @date 2015-5-7
 * @author salmon
 */

#ifndef CORE_MESH_TEST_MESH_TEST_H_
#define CORE_MESH_TEST_MESH_TEST_H_

#include <gtest/gtest.h>
#include <tuple>

#include "utilities.h"
#include "../../field/field.h"
#include "../structured/structured.h"
#include "../../io/io.h"

using namespace simpla;

typedef CartesianRectMesh mesh_type;

typedef typename mesh_type::coordinate_tuple coordinate_tuple;

class MeshTest: public testing::TestWithParam<
		std::tuple<size_t, nTuple<Real, 3>, nTuple<Real, 3>, nTuple<size_t, 3> > >
{
protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);

		std::tie(nid, xmin, xmax, dims) = GetParam();

		for (int i = 0; i < ndims; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
			{
				dims[i] = 1;
				xmax[i] = xmin[i];
			}
		}

		mesh.dimensions(&dims[0]);
		mesh.extents(xmin, xmax);
		mesh.deploy();

	}
	void TearDown()
	{

	}
public:
	typedef Real value_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinate_tuple coordinate_tuple;
	typedef typename mesh_type::topology_type topology_type;

	static constexpr size_t ndims = mesh_type::ndims;
	nTuple<Real, 3> xmin;
	nTuple<Real, 3> xmax;
	nTuple<size_t, 3> dims;
	value_type one;
	Real error;
	size_t nid;
	mesh_type mesh;

	virtual ~MeshTest()
	{
	}
};
TEST_P(MeshTest, foreach_hash)
{

	size_t count = 0;

	size_t max_num = NProduct(dims)
			* ((nid == 0 /*VERTEX*/|| nid == 7 /* VOLUME*/) ? 1 : 3);

	auto it = mesh.range(nid).begin();

	for (auto s : mesh.range(nid))
	{
		auto x = mesh.coordinates(s);

		EXPECT_GE(inner_product(x-xmin,xmax-x),0) << x << xmin << xmax; // point in box

		EXPECT_EQ(mesh.hash(s), count);
		++count;
	}
	EXPECT_EQ(count, max_num);

}
TEST_P(MeshTest, id)
{

	for (auto s : mesh.range(nid))
	{
		EXPECT_EQ(topology_type::pack(topology_type::unpack(s)), s);
	}

}

TEST_P(MeshTest, coordinates)
{

	size_t max_num = NProduct(dims)
			* ((nid == 0 /*VERTEX*/|| nid == 7 /* VOLUME*/) ? 1 : 3);

	for (auto s : mesh.range(nid))
	{

		EXPECT_EQ(topology_type::pack(topology_type::coordinates(s)), s);

		EXPECT_LE(
				abs(
						mesh.coordinates_from_topology(
								mesh.coordinates_to_topology(
										mesh.coordinates(s)))
								- mesh.coordinates(s)), abs(mesh.epsilon()));

		auto x = mesh.coordinates_global_to_local(mesh.coordinates(s), nid);

//		EXPECT_EQ( manifold.pack( manifold.coordinates_to_topology(manifold.coordinates(s)) ),s)
//				<< manifold.coordinates(s) << " "
//				<< manifold.coordinates_to_topology(manifold.coordinates(s)) << " "
//				<< std::hex << manifold.unpack(s);

	}

}
#endif /* CORE_MESH_TEST_MESH_TEST_H_ */
