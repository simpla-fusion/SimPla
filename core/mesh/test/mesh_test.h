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
TEST_P(MeshTest, foreach)
{

	CHECK(mesh.extents());

	for (auto s : mesh.domain<EDGE>())
	{
		SHOW(mesh.coordinates(s));
		SHOW(mesh.topology_type::coordinates(s));
	}

}
#endif /* CORE_MESH_TEST_MESH_TEST_H_ */
