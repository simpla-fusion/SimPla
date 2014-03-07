/*
 * rect_mesh_test.cpp
 *
 *  Created on: 2014年3月7日
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include "../fetl/fetl.h"
#include "../io/data_stream.h"
#include "../utilities/log.h"

#include "rect_mesh.h"

#define DEF_MESH RectMesh<>

using namespace simpla;

template<typename TI>
class TestMesh: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);
		mesh.SetDt(1.0);

		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 30, 10, 10 };
		mesh.SetDimensions(dims);

		mesh.Update();
	}
public:
	typedef RectMesh<> mesh_type;
	static constexpr int IForm = TI::value;
	mesh_type mesh;

	DEFINE_FIELDS(mesh_type)

};
typedef testing::Types<
//		Int2Type<VERTEX>,
        Int2Type<EDGE> /*, Int2Type<FACE>, Int2Type<VOLUME>*/> FormList;

TYPED_TEST_CASE(TestMesh, FormList);

TYPED_TEST(TestMesh, traversal){
{
	size_t count = 0;

	CHECK( TestFixture::mesh.GetDimensions());

	CHECK(TestFixture::mesh.GetNumOfElements(TestFixture::IForm ));

	auto s=*(TestFixture::mesh.template begin<TestFixture::IForm>());

	CHECK_BIT( s.d);
	s=TestFixture::mesh._R(s);
	CHECK_BIT( s.d);
	s=TestFixture::mesh._R(s);
	CHECK_BIT( s.d);

//	TestFixture::mesh.template Traversal < TestFixture::IForm > (
//			[& ](typename TestFixture::index_type s )
//			{	CHECK_BIT(TestFixture::mesh.s.d); ++count;}
//	);

	EXPECT_EQ(count,TestFixture:: mesh.GetNumOfElements( TestFixture::IForm));
}}
