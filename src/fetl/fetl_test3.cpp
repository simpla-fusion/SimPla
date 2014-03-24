/*
 * fetl_test3.cpp
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#include "fetl_test.h"
#include "../io/data_stream.h"
#include "save_field.h"

using namespace simpla;

DEFINE_FIELDS(DEF_MESH)

class TestFETLDiffCalcuate1: public testing::Test
{

protected:
	virtual void SetUp()
	{
		nTuple<3, Real> xmin = { -1.0, 1.0, 0 };
		nTuple<3, Real> xmax = { 2.0, 3.0, 4.0 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 1, 1, 32 };
		mesh.SetDimensions(dims);

		mesh.Update();

		auto const & dims_ = mesh.GetDimensions();

		for (int i = 0; i < 3; ++i)
		{
			if (dims_[i] <= 1)
				k[i] = 0;
		}

		dx = mesh.GetDx();

		k2 = Dot(k, k);

		CHECK(dims_);
		CHECK(mesh.strides_);
		CHECK(mesh.GetDx());

		GLOBAL_DATA_STREAM.OpenFile("FetlTest");
		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	Mesh mesh;

	typedef Real value_type;
	typedef typename Mesh::index_type index_type;
	typedef Field<Mesh, VERTEX, value_type> TZeroForm;
	typedef Field<Mesh, EDGE, value_type> TOneForm;
	typedef Field<Mesh, FACE, value_type> TTwoForm;
	typedef Field<Mesh, VOLUME, value_type> TThreeForm;

	static constexpr double PI = 3.141592653589793;
	nTuple<3, Real> k =
	{	2.0 * PI, 3 * PI, 4 * PI};
	coordinates_type dx;
	Real k2;
	static constexpr value_type default_value = 1.0;

};

TEST_F(TestFETLDiffCalcuate1, grad)
{
	TOneForm vf1(mesh);

	TZeroForm sf(mesh), sf2(mesh);

	sf.Clear();
	vf1.Clear();
	sf2.Clear();

	Real m = 0.0;
	Real k2 = Dot(k, k);

	Traversal<VERTEX>(mesh, [&]( index_type s )
	{
		CHECK(mesh.GetCoordinates(s));
		sf[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	});

	LOG_CMD(vf1 = Grad(sf));

	Real variance = 0;
	Real average = 0.0;

	Traversal<EDGE>(mesh, [&]( index_type s)
	{	auto x=mesh.GetCoordinates(s);

		auto expect= std::cos(Dot(k, x))*k[mesh._C(s)] * mesh.Volume(s);

		auto error = 0.5*k[mesh._C(s)]*k[mesh._C(s)] * mesh.Volume(s)*mesh.Volume(s);

		variance+= abs( (vf1[s]-expect)*(vf1[s]-expect));

		average+= (vf1[s]-expect);

		if((vf1[s]+expect) !=0 && error>0)
		EXPECT_LE(abs(2.0*(vf1[s]-expect)/(vf1[s] + expect)), error ) << vf1[s]<<" "<<expect<<" "<<x;

	});

	variance /= sf.size();
	average /= sf.size();
	CHECK(variance);
	CHECK(average);
	sf2 = Diverge(vf1) / k2;

//	auto rel_error2 = Dot(k, dx) * Dot(k, dx);
//
//	Traversal<VERTEX>(mesh, [&]( index_type s )
//	{
//		if(sf[s]!=0)
//		{
//			EXPECT_LE(abs(2.0*(sf[s]-sf2[s])/(sf[s]+sf2[s])) ,rel_error2)<< (sf[s])<<" "<< (sf2[s]);
//		}
//		else
//		{
//			EXPECT_LE(abs(sf2[s]) ,rel_error2);
//		}
//	});

	LOGGER << DUMP(sf);
	LOGGER << DUMP(vf1);
	LOGGER << DUMP(sf2);

}
//
//TEST_F(TestFETLDiffCalcuate1, curl1)
//{
//	typedef TOneForm TFA;
//	typedef TTwoForm TFB;
//	TFA vfa(mesh);
//	TFB vfb(mesh);
//
//	vfa.Clear();
//	vfb.Clear();
//
//	auto error = Dot(k, dx);
//
//	Traversal<TFA::IForm>(mesh, [&]( index_type s )
//	{
//		auto n=mesh._C(s);
//
//		auto a=Dot(mesh.GetCoordinates(s),k);
//
//		switch(n)
//		{
//			case 0:
//			vfa[s]= std::sin(a);
//			break;
//			case 1:
//			vfa[s]= std::cos(a);
//			break;
//			case 2:
//			vfa[s]= std::sin(a)*std::cos(a);
//			break;
//		}
//
//	});
//
//	vfb = Curl(vfa);
//
//	Real variance = 0;
//	Real average = 0.0;
//
//	Traversal<TFB::IForm>(mesh, [&]( index_type s )
//	{
//
//		value_type expect2;
//		auto n=mesh._C(s);
//
//		auto a=Dot(mesh.GetCoordinates(s),k);
//
//		nTuple<3,value_type> expect=
//		{
//			(2*std::cos(a)*std::cos(a)-1)*k[1],
//
//			-(2*std::cos(a)*std::cos(a)-1)*k[0],
//
//			-std::sin(a)*k[0] -std::cos(a)*k[1]
//		};
//
//		expect*=mesh.Volume(s);
//
//		variance+= abs( (vfb[s]-expect[n])*(vfb[s]-expect[n]));
//
//		average+= (vfb[s]-expect[n]);
//
//		if(expect[n] !=0)
//		{
//			EXPECT_LE( 2.0*abs((vfb[s]-expect[n])/(vfb[s]+expect[n])) ,error )
//			<< vfb[s]<<" "<<expect <<" "<<n<<" "<<(mesh.I(s)>>4)<<" "<<(mesh.J(s)>>4);
//		}
//		else
//		{
//			EXPECT_LE( abs(vfb[s]) ,error );
//		}
//
//	});
//
//	variance /= vfb.size();
//	average /= vfb.size();
//	CHECK(variance);
//	CHECK(average);
//
//////#ifdef DEBUG
////	LOGGER << DUMP(vfa);
////	LOGGER << DUMP(vfb);
//////#endif
//
//}

//template<typename TP>
//class TestFETLDiffCalcuate: public testing::Test
//{
//
//protected:
//	virtual void SetUp()
//	{
//		nTuple<3, Real> xmin = { 0, 0, 0 };
//		nTuple<3, Real> xmax = { 2, 3, 4 };
//		mesh.SetExtent(xmin, xmax);
//
//		nTuple<3, size_t> dims = { 16, 0, 0 };
//		mesh.SetDimensions(dims);
//
//		mesh.Update();
//
//		SetValue(&default_value);
//	}
//public:
//
//	Mesh mesh;
//
//	typedef TP value_type;
//	typedef typename Mesh::index_type index_type;
//	typedef Field<Mesh, VERTEX, value_type> TZeroForm;
//	typedef Field<Mesh, EDGE, value_type> TOneForm;
//	typedef Field<Mesh, FACE, value_type> TTwoForm;
//	typedef Field<Mesh, VOLUME, value_type> TThreeForm;
//
//	static constexpr double PI = 3.141592653589793;
//	static constexpr nTuple<3, Real> k = { 2.0 * PI, 3 * PI, 0.0 };
//
//	double RelativeError(double a, double b)
//	{
//		return (2.0 * fabs((a - b) / (a + b)));
//	}
//
//	void SetValue(double *v)
//	{
//		*v = 1.0;
//	}
//
//	void SetValue(Complex *v)
//	{
//		*v = Complex(1.0, 2.0);
//	}
//
//	template<int N, typename TV>
//	void SetValue(nTuple<N, TV> *v)
//	{
//		for (size_t i = 0; i < N; ++i)
//		{
//			SetValue(&((*v)[i]));
//		}
//	}
//
//	value_type default_value;
//
//};
//
//typedef testing::Types<double, Complex, nTuple<3, double> > PrimitiveTypes;
//
//TYPED_TEST_CASE(TestFETLDiffCalcuate, PrimitiveTypes);
//
//TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_0){
//{
//	Mesh const & mesh = TestFixture::mesh;
//
//	typename TestFixture::value_type v;
//
//	TestFixture::SetValue(&v);
//
//	typename TestFixture::TOneForm vf1(mesh,v);
//	typename TestFixture::TOneForm vf1b(mesh,v);
//	typename TestFixture::TTwoForm vf2(mesh,v);
//	typename TestFixture::TTwoForm vf2b(mesh,v);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	typename TestFixture::TZeroForm sf(mesh,v);
//
//	Real m=0.0;
//
//	for(auto & p:sf)
//	{
//		p = uniform_dist(gen);
//		m+= abs(p);
//	}
//
//	m/=sf.size();
//
//	LOG_CMD(vf1 = Grad(sf));
//	LOG_CMD(vf2 = Curl(vf1));
//	LOG_CMD(vf2b = Curl(Grad(sf)));
//
//	size_t count=0;
//	Real relative_error=0;
//
//	mesh.Traversal<FACE>(
//			[&](typename TestFixture::TTwoForm::index_type s)
//			{	relative_error+=abs(vf2[s]);
//
//				if(abs(vf2[s])>1.0e-10)
//				{
//					++count;
//				}
//			}
//
//	);
//	relative_error=relative_error/m;
//
//	INFORM2(relative_error);
//
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;
//
//	count =0;
//	relative_error=0.0;
//
//	mesh.Traversal<FACE>(
//			[&](typename TestFixture::TTwoForm::index_type s)
//			{	relative_error+=abs(vf2b[s]);
//
//				if(abs(vf2b[s])>1.0e-10)
//				{
//					++count;
//				}
//			}
//	);
//	relative_error=relative_error/m;
//
//	INFORM2(relative_error);
//
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;
//
//}
//}
//
//TYPED_TEST(TestFETLDiffCalcuate, curl_grad_eq_1){
//{
//	Mesh const & mesh = TestFixture::mesh;
//
//	typename TestFixture::value_type v;
//
//	TestFixture::SetValue(&v);
//
//	typename TestFixture::TOneForm vf1(mesh,v);
//	typename TestFixture::TOneForm vf1b(mesh,v);
//	typename TestFixture::TTwoForm vf2(mesh,v);
//	typename TestFixture::TTwoForm vf2b(mesh,v);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	typename TestFixture::TThreeForm vf(mesh,v);
//
//	Real m=0.0;
//
//	for(auto & p:vf)
//	{
//		p = uniform_dist(gen);
//		m+= abs(p);
//	}
//
//	m/=vf.size();
//
//	LOG_CMD(vf2 = Grad(vf));
//	LOG_CMD(vf1 = Curl(vf2));
//	LOG_CMD(vf1b = Curl(Grad(vf)));
//
//	size_t count=0;
//	Real relative_error=0;
//
//	mesh.Traversal<EDGE>(
//			[&](typename TestFixture::TTwoForm::index_type s)
//			{
//				relative_error+=abs(vf1[s]);
//
//				if(abs(vf1[s])>1.0e-10)
//				{
//					++count;
//				}
//			}
//
//	);
//	relative_error=relative_error/m;
//
//	INFORM2(relative_error);
//
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;
//
//	count =0;
//	relative_error=0.0;
//
//	mesh.Traversal<EDGE>(
//			[&](typename TestFixture::TTwoForm::index_type s)
//			{
//
//				relative_error+=abs(vf1b[s]);
//
//				if(abs(vf1b[s])>1.0e-10)
//				{
//					++count;
//				}
//			}
//
//	);
//	relative_error=relative_error/m;
//
//	INFORM2(relative_error);
//
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;
//
//}
//}
//TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_0){
//{
//
//	Mesh const & mesh = TestFixture::mesh;
//
//	typename TestFixture::value_type v;
//
//	v=1.0;
//
//	typename TestFixture::TZeroForm sf1(mesh);
//	typename TestFixture::TZeroForm sf2(mesh);
//	typename TestFixture::TOneForm vf1(mesh);
//	typename TestFixture::TTwoForm vf2(mesh,v);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	vf2.Init();
//
//	for(auto &p:vf2)
//	{
//		p*= uniform_dist(gen);
//	}
//
//	LOG_CMD(vf1 = Curl(vf2));
//
//	LOG_CMD(sf1 = Diverge( vf1));
//
//	LOG_CMD(sf2 = Diverge( Curl(vf2)));
//
//	size_t count=0;
//
//	Real m=0.0;
//
//	for(auto const &p:vf2)
//	{
//		m+=abs(p);
//	}
//	m/=vf2.size();
//
//	Real relative_error=0;
//	size_t num=0;
//	mesh.Traversal<VERTEX>(
//			[&](typename TestFixture::TZeroForm::index_type s)
//			{
//				relative_error+=abs(sf1[s]);
//
//				if(abs(sf1[s])>1.0e-10*m)
//				{
//					++count;
//				}
//			}
//	);
//
//	relative_error=relative_error/m;
//	INFORM2(relative_error);
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;
//
//	count =0;
//	relative_error=0.0;
//	mesh.Traversal<VERTEX>(
//			[&](typename TestFixture::TZeroForm::index_type s)
//			{
//				relative_error+=abs(sf2[s]);
//				if(abs(sf2[s])>1.0e-10*m)
//				{
//					++count;
//				}
//			}
//	);
//
//	relative_error=relative_error/m;
//	INFORM2(relative_error);
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;
//
//}
//}
//
//TYPED_TEST(TestFETLDiffCalcuate, div_curl_eq_1){
//{
//
//	Mesh const & mesh = TestFixture::mesh;
//
//	typename TestFixture::value_type v;
//
//	v=1.0;
//
//	typename TestFixture::TThreeForm sf1(mesh);
//	typename TestFixture::TThreeForm sf2(mesh);
//	typename TestFixture::TOneForm vf1(mesh);
//	typename TestFixture::TTwoForm vf2(mesh,v);
//
//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
//
//	vf1.Init();
//
//	for(auto &p:vf1)
//	{
//		p*= uniform_dist(gen);
//	}
//
//	LOG_CMD(vf2 = Curl(vf1));
//
//	LOG_CMD(sf1 = Diverge( vf2));
//
//	LOG_CMD(sf2 = Diverge( Curl(vf1)));
//
//	size_t count=0;
//
//	Real m=0.0;
//
//	for(auto const &p:vf2)
//	{
//		m+=abs(p);
//	}
//	m/=vf2.size();
//
//	Real relative_error=0;
//	size_t num=0;
//	mesh.Traversal<VOLUME>(
//			[&](typename TestFixture::TZeroForm::index_type s)
//			{
//				relative_error+=abs(sf1[s]);
//
//				if(abs(sf1[s])>1.0e-10*m)
//				{
//					++count;
//				}
//			}
//	);
//
//	relative_error=relative_error/m;
//	INFORM2(relative_error);
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;
//
//	count =0;
//	relative_error=0.0;
//	mesh.Traversal<VOLUME>(
//			[&](typename TestFixture::TZeroForm::index_type s)
//			{
//				relative_error+=abs(sf2[s]);
//				if(abs(sf2[s])>1.0e-10*m)
//				{
//					++count;
//				}
//			}
//	);
//
//	relative_error=relative_error/m;
//	INFORM2(relative_error);
//	EXPECT_GT(1.0e-8,relative_error);
//	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;
//
//}
//}
