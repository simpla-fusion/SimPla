/*
 * mesh_rectangle_test.cpp
 *
 *  Created on: 2014年3月7日
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include "../fetl/fetl.h"
#include "../io/data_stream.h"
#include "../utilities/log.h"

#include "octree_forest.h"
#include "mesh_rectangle.h"
#include "geometry_euclidean.h"
#include "traversal.h"
#define DEF_MESH RectMesh<>

using namespace simpla;

template<typename TI>
class TestMesh: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 17, 1, 1 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 1, 17, 1 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 1, 1, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 1, 17, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 17, 1, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 17, 17, 1 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 17, 17, 17 }));
		dims_list.emplace_back(nTuple<3, size_t>(
		{ 17, 33, 65 }));

	}
public:
	typedef OcForest mesh_type;
	static constexpr int IForm = TI::value;
	typedef typename OcForest::index_type index_type;
	std::vector<nTuple<3, size_t>> dims_list;

};

TYPED_TEST_CASE_P(TestMesh);

TYPED_TEST_P(TestMesh, traversal){
{
	for(auto const & dims:TestFixture::dims_list)
	{
		typename TestFixture::mesh_type mesh;

		mesh.SetDimensions(dims);

		mesh.Update();

		size_t count = 0;

		Traversal < TestFixture::IForm > ( mesh,
				[& ](typename TestFixture::index_type s )
				{
					++count;
				}
		);

		EXPECT_EQ(count, mesh.GetNumOfElements( TestFixture::IForm))<<mesh.GetDimensions();
	}

}}

typedef testing::Types<Int2Type<VERTEX>, Int2Type<EDGE>, Int2Type<FACE>, Int2Type<VOLUME> > FormList;

REGISTER_TYPED_TEST_CASE_P(TestMesh, traversal);

INSTANTIATE_TYPED_TEST_CASE_P(Mesh, TestMesh, FormList);
