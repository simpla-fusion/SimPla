/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
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

template<typename TF>
class TestFETLBasicArithmetic: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

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

		mesh.Update();

	}
public:
	typedef TF FieldType;
	typedef Form<0> RScalarField;

	Mesh mesh;

};

typedef testing::Types<Form<0>, Form<1>, Form<2>, Form<3>

, CForm<0>, CForm<1>, CForm<2>, CForm<3>

, VectorForm<0>, VectorForm<1>, VectorForm<2>, VectorForm<3>

> AllFieldTypes;

TYPED_TEST_CASE(TestFETLBasicArithmetic, AllFieldTypes);

TYPED_TEST(TestFETLBasicArithmetic,create_write_read){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::FieldType f( mesh );

	typename TestFixture::FieldType::value_type a; a= 1.0;

	f=0.0;

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

	typename TestFixture::FieldType f1(TestFixture::mesh),f2(TestFixture::mesh);

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type a; a = 3.0;

	f2 = a;
	f1 = 0;

	for (auto p : f2)
	{
		ASSERT_EQ(a,p)<<"idx="<< p;
	}

	size_t s=0;
	TestFixture::mesh.ForEach( [&](value_type & v)
			{	v=a*static_cast<Real>(s); ++s;},&f1 );

	f1 += f2;

	s=0;
	TestFixture::mesh.ForEach(

			[& ](typename TestFixture::FieldType::value_type v)
			{

				typename TestFixture::FieldType::value_type res;
				res=a+a*static_cast<Real>(s);
				ASSERT_EQ( res,v)<<s;
				++s;
			}
			,f1
	);

	f1*=2.0;

	s=0;
	TestFixture::mesh.ForEach(

			[& ](typename TestFixture::FieldType::value_type v)
			{

				typename TestFixture::FieldType::value_type res;
				res=(a+a*static_cast<Real>(s))*2.0;
				ASSERT_EQ( res,v)<<s;
				++s;
			},
			f1
	);
}
}
TYPED_TEST(TestFETLBasicArithmetic, constant_real){
{

	typename TestFixture::FieldType f1( TestFixture::mesh),f2(TestFixture::mesh),f3(TestFixture::mesh);

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type va,vb;

	va=2.0;vb=3.0;

	f1=va;
	f2=vb;

	f3 = - f1 *2.0 + f2*c - f1/b
	;

	TestFixture::mesh.ForEach(

			[&](typename TestFixture::FieldType::value_type v1,
					typename TestFixture::FieldType::value_type v2,
					typename TestFixture::FieldType::value_type v3)
			{
				value_type res;
				res= - v1*2.0 + v2 *c -v1/b;
				ASSERT_EQ( res, v3);
			},f1,f2,f3
	);
}
}

TYPED_TEST(TestFETLBasicArithmetic, scalar_field){
{
	//FIXME  should test with non-uniform field

	typedef typename TestFixture::FieldType::value_type value_type;

	typename TestFixture::FieldType f1( TestFixture::mesh),f2( TestFixture::mesh),
	f3( TestFixture::mesh),f4( TestFixture::mesh);

	typename TestFixture::RScalarField a( TestFixture::mesh);
	typename TestFixture::RScalarField b( TestFixture::mesh);
	typename TestFixture::RScalarField c( TestFixture::mesh);

	Real ra=1.0,rb=10.0,rc=100.0;
	typename TestFixture::FieldType::value_type va,vb,vc;

	va=ra;
	vb=rb;
	vc=rc;
	a=ra;
	b=rb;
	c=rc;
	f1=0.0;
	f2=0.0;
	f3=0.0;

	size_t count=0;

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	TestFixture::mesh.ForEach(

			[&](typename TestFixture::FieldType::value_type & v1 )
			{
				v1=va *uniform_dist(gen);

			}, &f1
	);

	TestFixture::mesh.ForEach(

			[&](typename TestFixture::FieldType::value_type & v1 )
			{
				v1=vb *uniform_dist(gen);

			},&f2
	);

	TestFixture::mesh.ForEach(

			[&](typename TestFixture::FieldType::value_type & v1 )
			{
				v1=vc *uniform_dist(gen);

			},&f3
	);

	f4= ( -f1*a +f2*b ) -f3/c -f1;

//	Plus( Minus(Negate(Wedge(f1,a)),Divides(f2,b)),Multiplies(f3,c) )
	;
	/**           (+)
	 *           /   \
	 *         (-)    (*)
	 *        /   \    | \
	 *      (^)    (/) f1 c
	 *     /  \   /  \
	 *-f1      a f2   b
	 *
	 * */
	count =0;

	TestFixture::mesh.ForEach(
			[&](value_type const &s1,value_type const &s2 ,
					value_type const &s3,value_type const &s4)
			{
				typename TestFixture::FieldType::value_type res;
				res=( -s1*ra +s2*rb ) -s3/rc -s1;

				if(res!=s4)
				{
					++count;
				}

			},f1,f2,f3,f4
	);

	EXPECT_EQ(0,count)<< "number of error points =" << count;

}
}

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

typedef testing::Types<double, Complex, nTuple<3, Real> > VecFieldTypes;

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

	typename TestFixture::VectorField va(mesh,vc2), vb(mesh), res_vector_field(
			mesh);

	res_scalar_field = Dot(vc1, va);

	res_vector_field = Cross( va,vc1);

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
			},res_vector_field

	);

}
}

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

		mesh.Update();

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

	void SetValue(double *v)
	{
		*v = 1.0;
	}

	void SetValue(Complex *v)
	{
		*v = Complex(1.0, 2.0);
	}

	template<int N, typename TV>
	void SetValue(nTuple<N, TV> *v)
	{
		for (size_t i = 0; i < N; ++i)
		{
			SetValue(&((*v)[i]));
		}
	}
};

typedef testing::Types<double, Complex, nTuple<3, double>, nTuple<3, nTuple<3, double>> > PrimitiveTypes;

TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);

TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::SetValue(&v);

	typename TestFixture::TZeroForm sf(mesh,v);
	typename TestFixture::TTwoForm vf2(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;

	for(auto & p:sf)
	{
		p = uniform_dist(gen);
		m+= abs(p);
	}

	m/=sf.size();

	vf2 = Curl(Grad(sf));

	size_t count=0;
	Real relative_error=0;
	mesh.ForEach(
			[&](typename TestFixture::TTwoForm::value_type const & u)
			{	relative_error+=abs(u);
				count+=( abs(u)>1.0e-10)?1:0;
			},
			vf2
	);
	relative_error=relative_error/m;
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

}
}

TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	v=1.0;

	typename TestFixture::TZeroForm sf(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto &p:vf2)
	{
		p*= uniform_dist(gen);
	}

	vf1 = Curl(vf2);
	sf = Diverge( Curl(vf2));

	size_t count=0;

	Real m=0.0;

	for(auto const &p:vf2)
	{
		m+=abs(p);
	}
	m/=vf2.size();

	Real relative_error=0;
	size_t num=0;
	mesh.ForEach(
			[&](typename TestFixture::TZeroForm::value_type const &s)
			{
				relative_error+=abs(s);
				count+=( abs(s)>1.0e-10*m)?1:0;
			},sf
	);

	relative_error=relative_error/m;
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

}
}
