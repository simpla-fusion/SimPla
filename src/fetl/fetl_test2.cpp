/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <random>

#include "fetl.h"

#include "../utilities/log.h"
#include "../mesh/co_rect_mesh.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;

DEFINE_FIELDS(CoRectMesh<>)

template<typename T>
class TestFETLVecAlgegbra: public testing::Test
{
protected:
	virtual void SetUp()
	{
		mesh.dt_ = 1.0;
		mesh.xmin_[0] = 0;
		mesh.xmin_[1] = 0;
		mesh.xmin_[2] = 0;
		mesh.xmax_[0] = 1.0;
		mesh.xmax_[1] = 1.0;
		mesh.xmax_[2] = 1.0;
		mesh.dims_[0] = 1;
		mesh.dims_[1] = 10;
		mesh.dims_[2] = 10;

		mesh.Update();
	}
public:
	Mesh mesh;
	typedef T value_type;
	typedef nTuple<3, value_type> Vec3;
	typedef Field<Geometry<Mesh, 0>, T> ScalarField;
	typedef Field<Geometry<Mesh, 0>, nTuple<3, T> > VectorField;
};

typedef testing::Types<double, Complex> VecFieldTypes;

TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);

TYPED_TEST(TestFETLVecAlgegbra,vec_0_form){
{
	const Mesh& mesh = TestFixture::mesh;

	typename TestFixture::Vec3 vc1 =
	{	1.0, 2.0, 3.0};

	typename TestFixture::Vec3 vc2 =
	{	-1.0, 4.0, 2.0};

	typename TestFixture::Vec3 res_vec;

	res_vec = Cross(vc2,vc1);

	typename TestFixture::value_type res_scalar;

	res_scalar = Dot(vc1, vc2);

	typename TestFixture::ScalarField res_scalar_field(mesh);

	typename TestFixture::VectorField va(mesh), vb(mesh),

	res_vector_field1(mesh),res_vector_field2(mesh);

	CHECK(Dot(vc1,vc2));
	va=vc2;
	vb=vc1;

	LOG_CMD(res_scalar_field = Dot(vc1, va));

	LOG_CMD(res_vector_field1 = Cross( va,vc1));

	LOG_CMD(res_vector_field2 = Cross( va,vb));

	mesh.ForEach (

			[&](typename TestFixture::ScalarField::value_type const & v)
			{
				ASSERT_EQ(res_scalar, v);
			},res_scalar_field

	);

	mesh.ForEach (

			[&](typename TestFixture::VectorField::value_type const & v)
			{
				ASSERT_EQ(res_vec , v);
			},res_vector_field1

	);

	mesh.ForEach (
			[&](typename TestFixture::VectorField::value_type const & v)
			{
				ASSERT_EQ(res_vec , v);
			},res_vector_field2
	);

}
}
