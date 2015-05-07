/**
 * @file mesh_test.h
 *
 * @date 2015年5月7日
 * @author salmon
 */

#ifndef CORE_MESH_TEST_MESH_TEST_H_
#define CORE_MESH_TEST_MESH_TEST_H_

#include <gtest/gtest.h>
#include <tuple>

#include "../../utilities/utilities.h"
#include "../../field/field.h"
#include "../structured/structured.h"
#include "../../io/io.h"

using namespace simpla;

typedef CartesianRectMesh mesh_type;

typedef typename mesh_type::coordinates_type coordinates_type;

class MeshTest: public testing::TestWithParam<
		std::tuple<nTuple<Real, 3>, nTuple<Real, 3>, nTuple<size_t, 3>,
				nTuple<Real, 3>> >
{
protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);
		std::tie(xmin, xmax, dims, K_real) = GetParam();
		K_imag = 0;

		for (int i = 0; i < ndims; ++i)
		{
			if (dims[i] <= 1 || (xmax[i] <= xmin[i]))
			{
				dims[i] = 1;
				K_real[i] = 0.0;
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
	typedef typename mesh_type::coordinates_type coordinates_type;

	static constexpr size_t ndims = mesh_type::ndims;
	nTuple<Real, 3> xmin;
	nTuple<Real, 3> xmax;
	nTuple<size_t, 3> dims;
	nTuple<Real, 3> K_real; // @NOTE must   k = n TWOPI, period condition
	nTuple<scalar_type, 3> K_imag;
	value_type one;
	Real error;

	mesh_type mesh;

	virtual ~MeshTest()
	{
	}
};
TEST_P(MeshTest, foreach_hash)
{
	std::set<mesh_type::id_type> nids = { 0, 1, 6, 7 };

	for (auto nid : nids)
	{

		size_t count = 0;

		size_t max_num = NProduct(dims)
				* ((nid == 0 /*VERTEX*/|| nid == 7 /* VOLUME*/) ? 1 : 3);

		for (auto s : mesh.range(nid))
		{
			auto x = mesh.coordinates(s);

			EXPECT_GE(inner_product(x-xmin,xmax-x),0) << x << xmin << xmax; // point in box

			EXPECT_EQ(mesh.hash(s), count);
			++count;
		}
		EXPECT_EQ(count, max_num);
	}

}
#endif /* CORE_MESH_TEST_MESH_TEST_H_ */
