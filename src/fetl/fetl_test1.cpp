/*
 * test_field.cpp
 *
 *  Created on: 2012-1-13
 *      Author: salmon
 */

#include "fetl_test.h"

using namespace simpla;

DEFINE_FIELDS(DEF_MESH)

template<typename TF>
class TestFETL: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 20, 0, 0 };
		mesh.SetDimensions(dims);
		mesh.Update();
	}
public:
	typedef TF FieldType;
	typedef Form<0> RScalarField;
	typedef Mesh mesh_type;
	mesh_type mesh;

};

typedef testing::Types<

Form<0>, Form<1>, Form<2>, Form<3>,

CForm<0>, CForm<1>, CForm<2>, CForm<3>,

VectorForm<0>, VectorForm<1>, VectorForm<2>, VectorForm<3>,

CVectorForm<0>, CVectorForm<1>, CVectorForm<2>, CVectorForm<3>,

TensorForm<0>, TensorForm<1>, TensorForm<2>, TensorForm<3>,

CTensorForm<0>, CTensorForm<1>, CTensorForm<2>, CTensorForm<3>

> AllFieldTypes;

TYPED_TEST_CASE(TestFETL, AllFieldTypes);

TYPED_TEST(TestFETL,create_write_read){
{

	Mesh const & mesh = TestFixture::mesh;

	typename TestFixture::FieldType f( mesh );

	typename TestFixture::FieldType::value_type a; a= 1.0;

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
		typename TestFixture::FieldType::value_type res;
		res=a* (s);
		EXPECT_EQ(res,v ) <<"s =" << s;
		s+=1.0;
	}

}
}

TYPED_TEST(TestFETL,assign){
{

	typename TestFixture::FieldType f1(TestFixture::mesh),f2(TestFixture::mesh);

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type a; a = 3.0;

	f1.Init();
	f2.Init();

	TestFixture::mesh.template Traversal<TestFixture::FieldType::IForm>(

			[&](typename Mesh::index_type const &s)
			{
				f1[s]=0;
				f2[s]=a;
			}
	);

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

TYPED_TEST(TestFETL, constant_real){
{

	typename TestFixture::FieldType f1( TestFixture::mesh),f2(TestFixture::mesh),f3(TestFixture::mesh);

	Real a,b,c;
	a=1.0,b=-2.0,c=3.0;

	typedef typename TestFixture::FieldType::value_type value_type;

	value_type va,vb;

	va=2.0;vb=3.0;

	f1=va;
	f2=vb;

	LOG_CMD(f3 = -f1*a +f2*c - f1/b -f1 );

	TestFixture::mesh.template Traversal<TestFixture::FieldType::IForm>(

			[&](typename TestFixture::FieldType::index_type s )
			{
				value_type res;
				res= - f1[s]*a + f2[s] *c -f1[s]/b-f1[s];
				ASSERT_EQ( res, f3[s]);
			}
	);
}
}

TYPED_TEST(TestFETL, scalar_field){
{
	//FIXME  should test with non-uniform field

	typedef typename TestFixture::FieldType::value_type value_type;
	typedef typename TestFixture::FieldType::index_type index_type;

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

	mesh.template Traversal< TestFixture::FieldType::IForm>(
			[&]( index_type s)
			{
				value_type res= - f1[s]*ra +f2[s]* rb -f3[s]/ rc -f1[s];

				EXPECT_DOUBLE_EQ( abs(res), abs(f4[s]))<< "s= "<<(s.d);
			}
	);

	EXPECT_EQ(0,count)<< "number of error points =" << count;

}
}

