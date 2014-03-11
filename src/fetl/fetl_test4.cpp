/*
 * fetl_test4.cpp
 *
 *  Created on: 2014年3月11日
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
		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 20, 1, 1 };
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

typedef testing::Types<double, Complex  > VecFieldTypes;

TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);

TYPED_TEST(TestFETLVecAlgegbra,vec_zero_form){
{
	const Mesh& mesh = TestFixture::mesh;

	typename TestFixture::Vec3 vc1 =
	{	1.0, 2.0, 3.0};

	typename TestFixture::Vec3 vc2 =
	{	-1.0, 4.0, 2.0};

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	typename TestFixture::ScalarField res_scalar_field(mesh);

	typename TestFixture::VectorField vaf(mesh), vbf(mesh),

	res_vector_field (mesh);

	vaf.Init();
	vbf.Init();

	for(auto & p:vaf)
	{
		p = vc1*uniform_dist(gen);
	}
	for(auto & p:vbf)
	{
		p =vc2*uniform_dist(gen);
	}

	LOG_CMD(res_vector_field = Cross( vaf,vbf) );

	mesh.template Traversal<VERTEX> (

			[&](typename TestFixture::VectorField::index_type s)
			{
				ASSERT_EQ(Cross(vaf[s],vbf[s]), res_vector_field [s]);

			}

	);

	LOG_CMD(res_scalar_field = Dot(vaf, vbf));

	mesh.template Traversal<VERTEX> (

			[&](typename TestFixture::VectorField::index_type s)
			{

				ASSERT_EQ(Dot(vaf[s],vbf[s]),res_scalar_field[s]);
			}

	);

}
}
