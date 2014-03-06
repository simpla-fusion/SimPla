/*
 * fetl_test2.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
using namespace simpla;
DEFINE_FIELDS(DEF_MESH)

template<typename T>
class TestFETLVecAlgegbra: public testing::Test
{
protected:
	virtual void SetUp()
	{
		mesh.SetDt(1.0);

		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 20, 0, 0 };
		mesh.SetDimensions(dims);

		mesh.Update();

	}
public:
	Mesh mesh;
	typedef T value_type;
	typedef nTuple<3, value_type> Vec3;
	typedef Field<Mesh, VERTEX, T> ScalarField;
	typedef Field<Mesh, VERTEX, nTuple<3, T> > VectorField;
};

typedef testing::Types<double
//		, Complex
> VecFieldTypes;

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

	typename TestFixture::ScalarField a(mesh);a.Init();

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto & p:a)
	{
		p = uniform_dist(gen);
	}

	typename TestFixture::ScalarField res_scalar_field(mesh);

	typename TestFixture::VectorField va(mesh), vb(mesh),

	res_vector_field1(mesh),res_vector_field2(mesh);

	CHECK(Dot(vc1,vc2));
	va=vc2;
	vb=vc1;

	CHECK(Cross(vc1,vc2));

	LOG_CMD(res_scalar_field = Dot(vc1, va));

	LOG_CMD(res_vector_field1 = Cross( va,vc1) );

//	LOG_CMD(res_vector_field2 = Cross( va,vb) );

//	mesh. Traversal<VERTEX> (
//
//			[&](typename TestFixture::VectorField::index_type s)
//			{
//				ASSERT_EQ(res_scalar,res_scalar_field[s]);
//			}
//
//	);
//
//	mesh.Traversal<VERTEX> (
//
//			[&](typename TestFixture::VectorField::index_type s)
//			{
//				typename TestFixture::Vec3 res;
//				res=res_vec*2*a[s];
//				ASSERT_EQ(res, res_vector_field1[s]);
//				ASSERT_EQ(res, res_vector_field1[s]);
//			}
//
//	);
}
}
