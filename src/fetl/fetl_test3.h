/*
 * fetl_test3.h
 *
 *  Created on: 2014年3月24日
 *      Author: salmon
 */

#ifndef FETL_TEST3_H_
#define FETL_TEST3_H_

#include <gtest/gtest.h>
#include <random>

#include "../io/data_stream.h"
#include "save_field.h"

#include "fetl.h"

#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;

template<typename TM>
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

//		GLOBAL_DATA_STREAM.OpenFile("FetlTest");
//		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	typedef TM mesh_type;
	typedef Real value_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;

	mesh_type mesh;

	static constexpr double PI = 3.141592653589793;

	nTuple<3, Real> k = { 2.0 * PI, 3 * PI, 4 * PI };

	coordinates_type dx;

	Real k2;

	static constexpr value_type default_value = 1.0;

};
TYPED_TEST_CASE_P(TestFETLDiffCalcuate1);

TYPED_TEST_P(TestFETLDiffCalcuate1, grad){
{
	typename TestFixture::mesh_type const & mesh=TestFixture::mesh;
	typedef typename TestFixture::index_type index_type;

	typename TestFixture::TOneForm vf1(mesh);

	typename TestFixture::TZeroForm sf(mesh), sf2(mesh);

	auto k=TestFixture::k;

	sf.Clear();
	vf1.Clear();
	sf2.Clear();

	Real m = 0.0;
	Real k2 = Dot(k, k);

	for(auto s :mesh.GetRegion( VERTEX))
	{
		sf[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf1 = Grad(sf));

	Real variance = 0;
	Real average = 0.0;

	for(auto s :mesh.GetRegion( EDGE))
	{	auto x=mesh.GetCoordinates(s);

		auto expect= std::cos(Dot(k, x))*k[mesh._C(s)] * mesh.Volume(s);

		auto error = 0.5*k[mesh._C(s)]*k[mesh._C(s)] * mesh.Volume(s)*mesh.Volume(s);

		variance+= abs( (vf1[s]-expect)*(vf1[s]-expect));

		average+= (vf1[s]-expect);

		if((vf1[s]+expect) !=0 && error>0)
		EXPECT_LE(abs(2.0*(vf1[s]-expect)/(vf1[s] + expect)), error ) << vf1[s]<<" "<<expect<<" "<<x;

	}

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
}

REGISTER_TYPED_TEST_CASE_P(TestFETLDiffCalcuate1, grad);

template<typename TF>
class TestFETLDiffCalcuate: public testing::Test
{

protected:
	virtual void SetUp()
	{
		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 2, 3, 4 };
		mesh.SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 16, 0, 0 };
		mesh.SetDimensions(dims);

		mesh.Update();

		SetValue(&default_value);
	}
public:

	typedef typename TF::mesh_type mesh_type;
	mesh_type mesh;

	typedef typename TF::value_type value_type;
	typedef typename mesh_type::index_type index_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;

	static constexpr double PI = 3.141592653589793;
	static constexpr nTuple<3, Real> k = { 2.0 * PI, 3 * PI, 0.0 };

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

	value_type default_value;

};

TYPED_TEST_CASE_P(TestFETLDiffCalcuate);

TYPED_TEST_P(TestFETLDiffCalcuate, curl_grad_eq_0){
{
	auto const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::SetValue(&v);

	typename TestFixture::TOneForm vf1(mesh,v);
	typename TestFixture::TOneForm vf1b(mesh,v);
	typename TestFixture::TTwoForm vf2(mesh,v);
	typename TestFixture::TTwoForm vf2b(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	typename TestFixture::TZeroForm sf(mesh,v);

	Real m=0.0;

//	for(auto & p:sf)
//	{
//		p = uniform_dist(gen);
//		m+= abs(p);
//	}

	for(auto s:mesh.GetRegion(VERTEX))
	{	sf[s]= uniform_dist(gen); m+=abs(sf[s]);}

	m/=sf.size();

	LOG_CMD(vf1 = Grad(sf));
	LOG_CMD(vf2 = Curl(vf1));
	LOG_CMD(vf2b = Curl(Grad(sf)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRegion(FACE))
	{
		relative_error+=abs(vf2[s]);

		if(abs(vf2[s])>1.0e-10)
		{
			++count;
		}
	}

	relative_error=relative_error/m;

	INFORM2(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

	count =0;
	relative_error=0.0;

	for(auto s:mesh.GetRegion(FACE))
	{
		relative_error+=abs(vf2b[s]);

		if(abs(vf2b[s])>1.0e-10)
		{
			++count;
		}
	}

	relative_error=relative_error/m;

	INFORM2(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

}
}

TYPED_TEST_P(TestFETLDiffCalcuate, curl_grad_eq_1){
{
	auto const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	TestFixture::SetValue(&v);

	typename TestFixture::TOneForm vf1(mesh,v);
	typename TestFixture::TOneForm vf1b(mesh,v);
	typename TestFixture::TTwoForm vf2(mesh,v);
	typename TestFixture::TTwoForm vf2b(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	typename TestFixture::TThreeForm vf(mesh,v);

	Real m=0.0;

//	for(auto & p:vf)
//	{
//		p = uniform_dist(gen);
//		m+= abs(p);
//	}
//
	for(auto s:mesh.GetRegion(VOLUME))
	{	vf[s]= uniform_dist(gen); m+=abs(vf[s]);}

	m/=vf.size();

	LOG_CMD(vf2 = Grad(vf));
	LOG_CMD(vf1 = Curl(vf2));
	LOG_CMD(vf1b = Curl(Grad(vf)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRegion(EDGE))
	{
		relative_error+=abs(vf1[s]);

		if(abs(vf1[s])>1.0e-10)
		{
			++count;
		}
	}

	relative_error=relative_error/m;

	INFORM2(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

	count =0;
	relative_error=0.0;

	for(auto s:mesh.GetRegion(EDGE))
	{

		relative_error+=abs(vf1b[s]);

		if(abs(vf1b[s])>1.0e-10)
		{
			++count;
		}
	}

	relative_error=relative_error/m;

	INFORM2(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< "number of non-zero points =" << count;

}
}
TYPED_TEST_P(TestFETLDiffCalcuate, div_curl_eq_0){
{

	auto const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	v=1.0;

	typename TestFixture::TZeroForm sf1(mesh);
	typename TestFixture::TZeroForm sf2(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	vf2.Init();
//
//	for(auto &p:vf2)
//	{
//		p*= uniform_dist(gen);
//	}

	Real m=0.0;

	for(auto s:mesh.GetRegion(FACE))
	{	vf2[s]=v* uniform_dist(gen); m+=abs(vf2[s]);}

	m/=vf2.size();
	LOG_CMD(vf1 = Curl(vf2));

	LOG_CMD(sf1 = Diverge( vf1));

	LOG_CMD(sf2 = Diverge( Curl(vf2)));

	size_t count=0;
	Real relative_error=0;
	size_t num=0;
	for(auto s:mesh.GetRegion(VERTEX))
	{
		relative_error+=abs(sf1[s]);

		if(abs(sf1[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

	count =0;
	relative_error=0.0;
	for(auto s:mesh.GetRegion(VERTEX))
	{
		relative_error+=abs(sf2[s]);
		if(abs(sf2[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

}
}

TYPED_TEST_P(TestFETLDiffCalcuate, div_curl_eq_1){
{

	auto const & mesh = TestFixture::mesh;

	typename TestFixture::value_type v;

	v=1.0;

	typename TestFixture::TThreeForm sf1(mesh);
	typename TestFixture::TThreeForm sf2(mesh);
	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh,v);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	vf1.Init();

//	for(auto &p:vf1)
//	{
//		p*= uniform_dist(gen);
//	}

	Real m=0.0;
	for(auto s:mesh.GetRegion(EDGE))
	{	vf1[s]=v* uniform_dist(gen); m+=abs(vf1[s]);}

	m/=vf2.size();

	LOG_CMD(vf2 = Curl(vf1));

	LOG_CMD(sf1 = Diverge( vf2));

	LOG_CMD(sf2 = Diverge( Curl(vf1)));

	size_t count=0;

	m/=vf2.size();
	Real relative_error=0;
	size_t num=0;
	for(auto s:mesh.GetRegion(VOLUME))
	{
		relative_error+=abs(sf1[s]);

		if(abs(sf1[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

	count =0;
	relative_error=0.0;
	for(auto s:mesh.GetRegion(VOLUME))
	{
		relative_error+=abs(sf2[s]);
		if(abs(sf2[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< "number of non-zero points =" << count;

}
}

REGISTER_TYPED_TEST_CASE_P(TestFETLDiffCalcuate, div_curl_eq_0, div_curl_eq_1, curl_grad_eq_0, curl_grad_eq_1);

#endif /* FETL_TEST3_H_ */
