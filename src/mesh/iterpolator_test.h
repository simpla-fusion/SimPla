/*
 * iterpolator_test.h
 *
 *  Created on: 2014年6月29日
 *      Author: salmon
 */

#ifndef ITERPOLATOR_TEST_H_
#define ITERPOLATOR_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../physics/constants.h"
#include "../io/data_stream.h"
#include "../parallel/message_comm.h"

template<typename TMesh>
class TestIterpolator: public testing::Test
{
protected:
	void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || xmax[i] <= xmin[i])
			{
				xmax[i] = xmin[i];
				dims[i] = 1;
			}
		}

		mesh.SetExtents(xmin, xmax, dims);

	}
public:
	typedef TMesh mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename TMesh::coordinates_type coordinates_type;

	unsigned int NDIMS = TMesh::NDIMS;

	mesh_type mesh;

	std::vector<unsigned int> iform_list =
	{ VERTEX, EDGE, FACE, VOLUME };

	coordinates_type xmin, xmax;

	nTuple<TMesh::NDIMS, index_type> dims;

};

TYPED_TEST_CASE_P(TestMesh);

#endif /* ITERPOLATOR_TEST_H_ */
