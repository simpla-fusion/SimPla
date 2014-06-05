/*
 * mesh_test2.cpp
 *
 *  Created on: 2014年6月5日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "mesh.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../parallel/message_comm.h"
using namespace simpla;

typedef Mesh<EuclideanGeometry<OcForest>> TMesh;

class TestMesh2: public testing::TestWithParam<

std::tuple<nTuple<TMesh::NDIMS, size_t>,

typename TMesh::coordinates_type,

typename TMesh::coordinates_type> >
{
protected:
	virtual void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

		auto param = GetParam();

		dims=std::get<0>(param);

		xmin=std::get<1>(param);

		xmax=std::get<2>(param);

		mesh.SetExtents(std::get<0>(param),std::get<1>(param), std::get<2>(param));

	}
public:
	typedef TMesh mesh_type;
	typedef typename mesh_type::Range Range;
	typedef typename Range::iterator iterator;
	static constexpr unsigned int NDIMS=TMesh::NDIMS;

	mesh_type mesh;

	std::vector<unsigned int> iforms =
	{	VERTEX, EDGE, FACE, VOLUME};

	typename TMesh::coordinates_type xmin,xmax;
	nTuple<TMesh::NDIMS, size_t> dims;
};

TEST_P(TestMesh2,Coordinates)
{
	mesh_type::coordinates_type x;

	x = (xmax + xmin) * 0.5;

	auto r = x;

	CHECK(r);

	auto idx = mesh.CoordinatesGlobalToLocal(&r);

	CHECK(mesh.Decompact(idx.self_));

	CHECK(r);

	EXPECT_EQ(mesh.CoordinatesLocalToGlobal(idx, r), x);
}

INSTANTIATE_TEST_CASE_P(SimPla, TestMesh2,

testing::Combine(testing::Values(nTuple<3, size_t>( { 13, 16, 10 })),

testing::Values(nTuple<3, Real>( { 0.0, 0.0, 0.0, })),

testing::Values(nTuple<3, Real>( { 1.0, 2.0, 2.0 }))));
