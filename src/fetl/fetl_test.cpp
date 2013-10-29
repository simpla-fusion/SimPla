/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "utilities/log.h"

#include "fetl.h"
#include "fetl/primitives.h"
#include "fetl/ntuple.h"
#include "fetl/expression.h"
#include "fetl/vector_calculus.h"
#include "physics/constants.h"

#include "mesh/uniform_rect.h"

using namespace simpla;

DEFINE_FIELDS(UniformRectMesh<3>)

template<typename TF>
class TestFETLBasicArithmetic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Log::Verbose(10);

		mesh.dt_ = 1.0;
		mesh.xmin_[0] = 0;
		mesh.xmin_[1] = 0;
		mesh.xmin_[2] = 0;
		mesh.xmax_[0] = 1.0;
		mesh.xmax_[1] = 1.0;
		mesh.xmax_[2] = 1.0;
		mesh.dims_[0] = 20;
		mesh.dims_[1] = 30;
		mesh.dims_[2] = 40;
		mesh.gw_[0] = 2;
		mesh.gw_[1] = 2;
		mesh.gw_[2] = 2;

		mesh.Init();

	}
public:
	typedef TF FieldType;

	Mesh mesh;

};

typedef testing::Types<RZeroForm, ROneForm, RTwoForm, RThreeForm, CZeroForm,
		COneForm, CTwoForm, CThreeForm, VecZeroForm, VecOneForm, VecTwoForm
//		,VecThreeForm
//		,CVecZeroForm, CVecOneForm, CVecTwoForm,CVecThreeForm
> AllFieldTypes;

//, VecThreeForm

// test arithmetic.h
TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);

TYPED_TEST(TestFETLBasicArithmetic,create_write_read){
{

	Log::Verbose(10);

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::FieldType f( mesh );

	typename TestFixture::FieldType::value_type a; a= 1.0;

	for (size_t s = 0, e=f.size(); s < e; ++s)
	{
		f[s] = a*static_cast<double>(s);
	}

	for (size_t s = 0, e=f.size(); s < e; ++s)
	{
		typename TestFixture::FieldType::value_type res;
		res=a*static_cast<Real>(s);
		ASSERT_EQ(res,f[s])<<"idx=" << s;
	}

}
}

TYPED_TEST(TestFETLBasicArithmetic,assign){
{
	typename TestFixture::FieldType::geometry_type geometry(TestFixture::mesh);

	typename TestFixture::FieldType f1(TestFixture::mesh),f2(TestFixture::mesh);

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type a; a = 3.0;

	std::fill(f1.begin(),f1.end(), a);

	for (auto p : f1)
	{
		ASSERT_EQ(a,p)<<"idx="<< p;
	}

	geometry.ForEach(
			[&f2,&a](typename TestFixture::FieldType::geometry_type::index_type const & s)
			{
				f2[s]=a*static_cast<Real>(s);
			}
	)
	;
	f1 += f2;

	geometry.ForEach(

			[&a,&f1](typename TestFixture::FieldType::geometry_type::index_type const & s)
			{
				typename TestFixture::FieldType::value_type res;
				res=a+a*static_cast<Real>(s);
				ASSERT_EQ( res,f1[s])<<"idx="<< s;
			}

	);

	f1*=2.0;

	geometry.ForEach(

			[&a,&f1](typename TestFixture::FieldType::geometry_type::index_type const & s)
			{
				typename TestFixture::FieldType::value_type res;
				res=(a+a*static_cast<Real>(s))*2.0;
				ASSERT_EQ( res,f1[s])<<"idx="<< s;
			}

	);
}
}
TYPED_TEST(TestFETLBasicArithmetic, constant_real){
{
	typename TestFixture::FieldType::geometry_type geometry(TestFixture::mesh);

	typename TestFixture::FieldType f1( TestFixture::mesh),f2(TestFixture::mesh),f3(TestFixture::mesh);

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type va,vb;

	va=2.0;vb=3.0;

	std::fill(f1.begin(),f1.end(), va);
	std::fill(f2.begin(),f2.end(), vb);

	f3 = -f1 *2.0
	+ f2 * c
	-f1/b
	;

	geometry.ForEach (

			[&](typename TestFixture::FieldType::geometry_type::index_type const & s)
			{
				value_type res;
				res= - f1[s]*2.0 + f2[s] *c -f1[s]/b
				;
				ASSERT_EQ( res, f3[s]) << s;
			}
	);
}
}

////TYPED_TEST(TestFETLBasicArithmetic, scalar_field){
////{
////	//FIXME  should test with non-uniform field
////
////	typename TestFixture::FieldType f1(typename TestFixture::mesh),f2(typename TestFixture::mesh),
////	f3(typename TestFixture::mesh),f4(typename TestFixture::mesh);
////
////	RScalarField a(typename TestFixture::mesh),b(typename TestFixture::mesh),c(typename TestFixture::mesh);
////
////	std::fill(a.begin(),a.end(), 1.0);
////	std::fill(b.begin(),b.end(), 3.0);
////	std::fill(c.begin(),c.end(), 5.0);
////
////	size_t count=0;
////
////	a.ForEach(
////			[&a,&b,&c,&count](size_t s)
////			{
////
////				if( 1.0 == a[*s] && 3.0== b[*s] && 5.0==c[*s])
////				{
////					++count;
////				}
////			}
////	);
////	EXPECT_EQ(a.size(),count);
////
////	typename TestFixture::FieldType::value_type va,vb,vc;
////
////	va=2.0;
////	vb=3.0;
////	vc=5.0;
////
////	f4= -(f1^a)-f2/b +f3*c;
//////	Plus( Minus(Negate(Wedge(f1,a)),Divides(f2,b)),Multiplies(f3,c) )
////	;
////	/**           (+)
////	 *           /   \
////	 *         (-)    (*)
////	 *        /   \    | \
////	 *      (^)    (/) f1 c
////	 *     /  \   /  \
////	 *-f1      a f2   b
////	 *
////	 * */
////	count =0;
////
////	size_t num_of_comp=f3.get_num_of_comp();
////
////	for (auto s = f3.get_center_elements_begin( );
////			s!=f3.get_center_elements_end( ); ++s)
////	{
////		typename TestFixture::FieldType::value_type res;
////		res=
////		-f1[*s]*a[*s/num_of_comp]
////		-f2[*s]/b[*s/num_of_comp]
////		+f3[*s]*c[*s/num_of_comp]
////		;
////
////		if(res==f4[*s])
////		{
////			++count;
////		}
////
//////		EXPECT_EQ(res,f4[*s])<<*s
//////		<<" "<<num_of_comp
//////		<<" "<<f1[*s]
//////		<<" "<<f2[*s]
//////		<<" "<<f3[*s]
//////		<<" "<<a[*s/num_of_comp]
//////		<<" "<<b[*s/num_of_comp]
//////		<<" "<<c[*s/num_of_comp]
//////		;
////
////	}
////	EXPECT_EQ(f3.get_num_of_center_elements(),count);
////
////}
////}
//// test vector_calculus.h
//template<typename T>
//class TestFETLVecAlgegbra: public testing::Test
//{
//protected:
//	virtual void SetUp()
//	{
//		mesh.dt = 1.0;
//		mesh.xmin[0] = 0;
//		mesh.xmin[1] = 0;
//		mesh.xmin[2] = 0;
//		mesh.xmax[0] = 1.0;
//		mesh.xmax[1] = 1.0;
//		mesh.xmax[2] = 1.0;
//		mesh.dims[0] = 20;
//		mesh.dims[1] = 30;
//		mesh.dims[2] = 40;
//		mesh.gw[0] = 2;
//		mesh.gw[1] = 2;
//		mesh.gw[2] = 2;
//
//		mesh.Init();
//	}
//public:
//	Mesh mesh;
//	typedef Field<Geometry<Mesh, 0>, T> ScalarField;
//	typedef Field<Geometry<Mesh, 0>, nTuple<3, T> > VectorField;
//};
//
//typedef testing::Types<double, Complex> VecFieldTypes;
//
//TYPED_TEST_CASE(TestFETLVecAlgegbra, VecFieldTypes);
//
////TYPED_TEST(TestFETLVecAlgegbra,constant_vector){
////{
////	const Mesh& mesh = TestFixture::mesh;
////
////	Geometry<Mesh, 0> geometry(TestFixture::mesh);
////
////	Vec3 vc1 =
////	{	1.0, 2.0, 3.0};
////	Vec3 vc2 =
////	{	-1.0, 4.0, 2.0};
////
////	Vec3 res_vec;
////
////	res_vec = Cross(vc1, vc2);
////
////	Real res_scalar;
////	res_scalar = Dot(vc1, vc2);
////
////	typename TestFixture::ScalarField res_scalar_field(mesh);
////
////	typename TestFixture::VectorField va(mesh), vb(mesh), res_vector_field(
////			mesh);
////
////	std::fill(va.begin(),va.end(), vc2);
////
////	res_scalar_field = Dot(vc1, va);
////
////	res_vector_field = Cross(vc1, va);
////
////	size_t num_of_comp = geometry.get_num_of_comp( );
////
////	for (auto s = geometry.get_center_elements_begin( );
////			s != geometry.get_center_elements_end( ); ++s)
////	{
////		EXPECT_EQ(res_scalar, res_scalar_field[(*s)] )<< "idx=" <<(*s)<< " | "
////		<<va[(*s)] <<" | "<< vc1 << " | "<< res_scalar_field[(*s)]
////		;
////
////		EXPECT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s)<< " | "
////		<<va[(*s)] <<" | "<< vc1 << " | "<< res_vector_field[(*s)]
////		;
////	}
////
////}
////}
////
////TYPED_TEST(TestFETLVecAlgegbra,complex_vector_field){
////{
////	//FIXME  should test with non-uniform field
////
////	mesh const & mesh = TestFixture::mesh;
////
////	Vec3 vc1 =
////	{	1.0,2.0,3.0};
////
////	CVec3 vc2 =
////	{
////		Complex( 0.0,0.0) ,
////		Complex( -0.2,0.2) ,
////		Complex( 3.0,1.3)};
////
////	Complex res_scalar= Dot(vc2,vc1);
////
////	CVec3 res_vec;
////
////	res_vec=Cross(vc1,vc2);
////
////	typename TestFixture::VectorField va(mesh);
////
////	typename TestFixture::CVectorField vb(mesh);
////
////	va = vc1;
////
////	vb = vc2;
////
////	typename TestFixture::CVectorField res_vector_field(mesh);
////	typename TestFixture::CScalarField res_scalar_field(mesh);
////
////	res_scalar_field = Dot(vb, va);
////
////	res_vector_field = Cross(va, vb);
////
////	size_t num_of_comp =mesh.get_num_of_comp(TestFixture::VectorField::IForm);
////
////	for (typename mesh::const_iterator s = mesh.get_center_elements_begin(TestFixture::VectorField::IForm);
////			s!=mesh.get_center_elements_end(TestFixture::VectorField::IForm); ++s)
////	{
////		ASSERT_EQ(res_scalar, res_scalar_field[(*s)] ) << "idx=" <<(*s);
////
////		ASSERT_EQ(res_vec, (res_vector_field[(*s)])) << "idx=" <<(*s);
////
////	}
////
////}
////}
////
template<typename TP>
class TestFETLDiffCalcuate: public testing::Test
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
		mesh.dims_[0] = 20;
		mesh.dims_[1] = 30;
		mesh.dims_[2] = 40;
		mesh.gw_[0] = 2;
		mesh.gw_[1] = 2;
		mesh.gw_[2] = 2;

		mesh.Init();

	}
public:

	Mesh mesh;

	typedef TP value_type;
	typedef Field<Geometry<Mesh, 0>, value_type> TZeroForm;
	typedef Field<Geometry<Mesh, 1>, value_type> TOneForm;
	typedef Field<Geometry<Mesh, 2>, value_type> TTwoForm;

	double RelativeError(double a, double b)
	{
		return (2.0 * fabs((a - b) / (a + b)));
	}

	void Setvalue_type(double *v)
	{
		*v = 1.0;
	}

	void Setvalue_type(Complex *v)
	{
		*v = Complex(1.0, 2.0);
	}

	template<int N, typename TV>
	void Setvalue_type(nTuple<N, TV> *v)
	{
		for (size_t i = 0; i < N; ++i)
		{
			Setvalue_type(&((*v)[i]));
		}
	}
};

typedef testing::Types<double, Complex, nTuple<3, double>,
		nTuple<3, nTuple<3, double> > > PrimitiveTypes;

TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);

TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::Setvalue_type(&v);

	typename TestFixture::TZeroForm sf(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh);

	mesh.ForEach(0,[&](typename Mesh::index_type const &s)
			{
				sf[s] = static_cast<double>(s)*v;
			}
	);

	vf1 = Grad(sf);

	vf2 = Curl(Grad(sf));

	mesh.ForEach(1,
			[&](typename Mesh::index_type const & s)
			{
				ASSERT_NE(0.0,abs(vf1[s])) << "idx=" << s;
			}
	);

	mesh.ForEach(2,
			[&](typename Mesh::index_type const & s)
			{
				ASSERT_DOUBLE_EQ(0.0,abs(vf2[s])) << "idx=" << s;
			}
	);
}
}

TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::TZeroForm sf(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh);

	typename TestFixture::value_type v;

	TestFixture::Setvalue_type(&v);

	mesh.ForEach(0,[&](typename Mesh::index_type const &s)
			{
				vf2[s] = static_cast<double>(s+1.0)*v;
//				CHECK(vf2[s]);
			}
	);

	vf1 = Curl(vf2);
	sf = Diverge( Curl(vf2));

	mesh.ForEach(0,
			[&sf](typename Mesh::index_type const &s)
			{
				ASSERT_TRUE(1.0e-20 > abs(sf[s])) << "idx=" << s;
			}
	);

}
}
