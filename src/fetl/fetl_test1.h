/*
 * fetl_test1.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST1_H_
#define FETL_TEST1_H_

#include <gtest/gtest.h>
#include <random>
#include "fetl.h"
#include "fetl_test.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;

template<typename TV, int IFORM>
struct TestFETLParam1
{
	typedef TMesh mesh_type;
	typedef TV value_type;
	static constexpr int IForm = IFORM;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin = { 1.0, 1.0, 1.0 };

		nTuple<3, Real> xmax = { 2.0, 3.0, 4.0 };

		nTuple<3, size_t> dims = { 16, 32, 67 };

		mesh->SetExtents(xmin, xmax, dims);

	}

	static void SetDefaultValue(value_type * v)
	{
		SetDefaultValue(v);
	}
	template<typename T>
	void SetDefaultValue(T* v)
	{
		*v = 1;
	}
	template<typename T>
	void SetDefaultValue(std::complex<T>* v)
	{
		T r;
		SetDefaultValue(&r);
		*v = std::complex<T>();
	}

	template<int N, typename T>
	void SetDefaultValue(nTuple<N, T>* v)
	{
		for (int i = 0; i < N; ++i)
		{
			(*v)[i] = i;
		}
	}
};

template<typename TParam>
class TestFETLBase: public testing::Test
{
protected:
	virtual void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

		TParam::SetUpMesh(&mesh);
		TParam::SetDefaultValue(&default_value);
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	static constexpr int IForm = TParam::IForm;

	typedef Field<mesh_type, VERTEX, Real> RScalarField;
	typedef Field<mesh_type, IForm, value_type> FieldType;

	mesh_type mesh;
	value_type default_value;

};

typedef testing::Types<

TestFETLParam1<Real, VERTEX>,

TestFETLParam1<Real, EDGE>,

TestFETLParam1<Real, FACE>,

TestFETLParam1<Real, VOLUME>,

TestFETLParam1<Complex, VERTEX>,

TestFETLParam1<Complex, EDGE>,

TestFETLParam1<Complex, FACE>,

TestFETLParam1<Complex, VOLUME>,

TestFETLParam1<nTuple<3, Real>, VERTEX>,

TestFETLParam1<nTuple<3, Real>, EDGE>,

TestFETLParam1<nTuple<3, Real>, FACE>,

TestFETLParam1<nTuple<3, Real>, VOLUME>,

TestFETLParam1<nTuple<3, Complex>, VERTEX>,

TestFETLParam1<nTuple<3, Complex>, EDGE>,

TestFETLParam1<nTuple<3, Complex>, FACE>,

TestFETLParam1<nTuple<3, Complex>, VOLUME>,

TestFETLParam1<nTuple<3, nTuple<3, Real>>, VERTEX>,

TestFETLParam1<nTuple<3, nTuple<3, Real>>, EDGE>,

TestFETLParam1<nTuple<3, nTuple<3, Real>>, FACE>,

TestFETLParam1<nTuple<3, nTuple<3, Real>>, VOLUME>,

TestFETLParam1<nTuple<3, nTuple<3, Complex>>, VERTEX>,

TestFETLParam1<nTuple<3, nTuple<3, Complex>>, EDGE>,

TestFETLParam1<nTuple<3, nTuple<3, Complex>>, FACE>,

TestFETLParam1<nTuple<3, nTuple<3, Complex>>, VOLUME>

> TypeParamList;

TYPED_TEST_CASE(TestFETLBase, TypeParamList);

TYPED_TEST(TestFETLBase,create_write_read){
{

	typename TestFixture::mesh_type const & mesh = TestFixture::mesh;

	typename TestFixture::FieldType f( mesh );

	typename TestFixture::value_type a; a= 1.0;

	f.Clear();
	f.Init();
	double s=0;

	for(auto & v : f )
	{
		v= a*(s);
		s+=1.0;
	}
	s=0;
	for(auto const & v : f )
	{
		typename TestFixture::value_type res;
		res=a* (s);
		EXPECT_EQ(res,v ) <<"s =" << s;
		s+=1.0;
	}

}
}

TYPED_TEST(TestFETLBase,assign){
{
	typename TestFixture::mesh_type const & mesh= TestFixture::mesh;

	typename TestFixture::FieldType f1(mesh),f2(mesh);

	typedef typename TestFixture::value_type value_type;

	value_type a; a = 3.0;

	f1.Init();
	f2.Init();

	for(auto s :mesh.Select( TestFixture::FieldType::IForm))
	{
		f1[s]=0;
		f2[s]=a;
	}

	for(value_type const & v : f2)
	{
		ASSERT_EQ(a,v)<<"v ="<< v;
	}

	for(value_type & v:f1)
	{
		v=a*2.0;
	}

	LOG_CMD(f1 += f2);

	value_type res;

	res=a+a*2.0;

	size_t count=0;

	for(auto v:f1)
	{
		count+=(res!=v?1:0);
	}

	ASSERT_EQ(count,0);

	LOG_CMD(f1*=2.0);

	res=(a+a*2.0)*2.0;

	for(auto v:f1)
	{
		ASSERT_EQ( res,v);
	}
}
}

TYPED_TEST(TestFETLBase, constant_real){
{
	typename TestFixture::mesh_type const & mesh= TestFixture::mesh;

	typename TestFixture::FieldType f1( mesh),f2(mesh),f3(mesh);

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type va,vb;

	va=2.0;vb=3.0;

	f1=va;
	f2=vb;

	LOG_CMD(f3 = -f1*a +f2*c - f1/b -f1 );

	for(auto s :mesh.Select( TestFixture::FieldType::IForm))
	{
		value_type res;
		res= - f1[s]*a + f2[s] *c -f1[s]/b-f1[s];
		ASSERT_EQ( res, f3[s]);
	}
}
}

TYPED_TEST(TestFETLBase, scalar_field){
{

	typedef typename TestFixture::FieldType::value_type value_type;

	typename TestFixture::mesh_type const &mesh=TestFixture::mesh;
	typename TestFixture::FieldType f1( mesh),f2( mesh),
	f3( mesh),f4( mesh);

	typename TestFixture::RScalarField a( mesh);
	typename TestFixture::RScalarField b( mesh);
	typename TestFixture::RScalarField c( mesh);

	Real ra=1.0,rb=10.0,rc=100.0;
	typename TestFixture::FieldType::value_type va,vb,vc;

	va=ra;
	vb=rb;
	vc=rc;

	a.Fill(ra);
	b.Fill(rb);
	c.Fill(rc);

	f1.Init();
	f2.Init();
	f3.Init();
	f4.Fill(0);

	size_t count=0;

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	for(auto & v:f1)
	{
		v=va *uniform_dist(gen);
	}
	for(auto & v:f2)
	{
		v=vb *uniform_dist(gen);
	}

	for(auto & v:f3)
	{
		v=vc *uniform_dist(gen);
	}

	LOG_CMD(f4= -f1*a +f2*b -f3/c -f1 );

	//	Plus( Minus(Negate(Wedge(f1,a)),Divides(f2,b)),Multiplies(f3,c) )

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

	for(auto s :mesh.Select( TestFixture::FieldType::IForm ) )
	{
		value_type res= - f1[s]*ra +f2[s]* rb -f3[s]/ rc -f1[s];

		EXPECT_LE( abs(res-f4[s]) ,1.0e-10 )<< "s= "<<(mesh.Hash(s));
	}

	EXPECT_EQ(0,count)<< "number of error points =" << count;

}
}

#endif /* FETL_TEST1_H_ */
