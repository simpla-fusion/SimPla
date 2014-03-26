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

template<typename TParam>
class TestDiffCalculus: public testing::Test
{

protected:
	virtual void SetUp()
	{
		TParam::SetUpMesh(&mesh);
		TParam::SetDefaultValue(&default_value);
		//		GLOBAL_DATA_STREAM.OpenFile("FetlTest");
		//		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;

	mesh_type mesh;

	static constexpr double PI = 3.141592653589793;

	static constexpr nTuple<3, Real> k = { 2.0 * PI, 1.0 * PI, 4.0 * PI };

	value_type default_value;

};
TYPED_TEST_CASE_P(TestDiffCalculus);

TYPED_TEST_P(TestDiffCalculus, grad){
{

	typedef typename TestFixture::index_type index_type;
	typedef typename TestFixture::value_type value_type;
	auto const & mesh= TestFixture::mesh;

	auto dx = mesh.GetDx();
	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	typename TestFixture::TOneForm vf1(mesh);

	typename TestFixture::TZeroForm sf(mesh), sf2(mesh);

	sf.Clear();
	vf1.Clear();
	sf2.Clear();

	for(auto s :mesh.GetRegion( VERTEX))
	{
		sf[s]= std::sin(Dot(k,mesh.GetCoordinates(s)))*mesh.Volume(s);
	};

	LOG_CMD(vf1 = Grad(sf));

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average*= 0.0;

	for(auto s :mesh.GetRegion( EDGE))
	{
		auto n=mesh._C(s);

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*k[n] * mesh.Volume(s);

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] ) * mesh.Volume(s)*mesh.Volume(s);

		variance+= abs( (vf1[s]-expect)*(vf1[s]-expect));

		average+=(vf1[s]-expect);

		if(abs(vf1[s])>1.0e-10|| abs(expect)>1.0e-10)
		EXPECT_LE(abs(2.0*(vf1[s]-expect)/(vf1[s] + expect)), error )<<" "<< vf1[s]<<" "<<expect;

	}

	variance /= sf.size();
	average /= sf.size();
	CHECK(variance);
	CHECK(average);

}
}
TYPED_TEST_P(TestDiffCalculus, diverge){
{
	typedef typename TestFixture::index_type index_type;
	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;

	auto dx = mesh.GetDx();
	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TTwoForm vf2(mesh);
	typename TestFixture::TThreeForm vf(mesh);
	typename TestFixture::TZeroForm sf(mesh);

	sf.Clear();
	vf1.Clear();

	for(auto s :mesh.GetRegion( EDGE))
	{
		vf1[s]= std::sin(Dot(k,mesh.GetCoordinates(s)))*mesh.Volume(s);
	};

	sf = Diverge(vf1);

	Real variance = 0;
	value_type average = 0.0;

	for(auto s :mesh.GetRegion( VERTEX))
	{

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*(k[0]+k[1]+k[2]) * mesh.Volume(s);

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] ) * dx[0]*dx[0];

		variance+= abs( (sf[s]-expect)*(sf[s]-expect));

		average+= (sf[s]-expect);

		if(abs(sf[s])>1.0e-10|| abs(expect)>1.0e-10)
		EXPECT_LE(abs(2.0*(sf[s]-expect)/(sf[s] + expect)), error )<<" "<< sf[s]<<" "<<expect;

	}

	variance /= sf.size();
	average /= sf.size();
	CHECK(variance);
	CHECK(average);

	vf2.Clear();
	vf.Clear();

	for(auto s :mesh.GetRegion( FACE))
	{
		vf2[s]= std::sin(Dot(k,mesh.GetCoordinates(s)))*mesh.Volume(s);
	};

	vf = Diverge(vf2);

	variance = 0;
	average = 0.0;

	for(auto s :mesh.GetRegion( VOLUME))
	{

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*(k[0]+k[1]+k[2]) * mesh.Volume(s);

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] ) * dx[0]*dx[0];

		variance+= abs( (vf[s]-expect)*(vf[s]-expect));

		average+= (vf[s]-expect);

		if(abs(vf[s])>1.0e-10|| abs(expect)>1.0e-10)
		EXPECT_LE(abs(2.0*(vf[s]-expect)/(vf[s] + expect)), error )<<" "<< vf[s]<<" "<<expect;

	}

	variance /= vf.size();
	average /= vf.size();
	CHECK(variance);
	CHECK(average);
}
}
TYPED_TEST_P(TestDiffCalculus, curl1){
{
	typedef typename TestFixture::index_type index_type;
	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;
	auto dx = mesh.GetDx();
	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TOneForm vf1b(mesh);
	typename TestFixture::TTwoForm vf2(mesh);
	typename TestFixture::TTwoForm vf2b(mesh);

	vf1.Clear();
	vf1b.Clear();
	vf2.Clear();
	vf2b.Clear();

	Real m = 0.0;
	Real variance = 0;
	value_type average; average*= 0.0;

	for(auto s :mesh.GetRegion(EDGE))
	{
		vf1[s]= std::sin(Dot(k,mesh.GetCoordinates(s)))*mesh.Volume(s);
	};

	LOG_CMD(vf2 = Curl(vf1));

	for(auto s :mesh.GetRegion( FACE))
	{
		auto n=mesh._C(s);

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*( k[(n+1)%3]- k[(n+2)%3] )* mesh.Volume(s);

		vf2b[s]=expect;

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] )* mesh.Volume(s);

		variance+= abs( (vf2[s]-expect)*(vf2[s]-expect));

		average+= (vf2[s]-expect);

		if( abs(vf2[s])>1.0e-10|| abs(expect)>1.0e-10)
		EXPECT_LE(abs(2.0*(vf2[s]-expect)/(vf2[s] + expect)), error )<<mesh.GetDimensions() << vf2[s]<<" "<<expect;

	}

	variance /= vf2.size();
	average /= vf2.size();
	CHECK(variance);
	CHECK(average);
	LOGGER<<DUMP(vf1);
	LOGGER<<DUMP(vf2);
	LOGGER<<DUMP(vf2b);

}
}

TYPED_TEST_P(TestDiffCalculus, curl2){
{
	typedef typename TestFixture::index_type index_type;
	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;

	auto dx = mesh.GetDx();
	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	typename TestFixture::TOneForm vf1(mesh);
	typename TestFixture::TOneForm vf1b(mesh);
	typename TestFixture::TTwoForm vf2(mesh);
	typename TestFixture::TTwoForm vf2b(mesh);

	vf1.Clear();
	vf1b.Clear();
	vf2.Clear();
	vf2b.Clear();

	Real m = 0.0;
	Real variance = 0;
	value_type average; average*= 0.0;

	for(auto s :mesh.GetRegion(FACE))
	{
		vf2[s]= std::sin(Dot(k,mesh.GetCoordinates(s)))*mesh.Volume(s);
	};

	LOG_CMD(vf1 = Curl(vf2));

	vf1b.Clear();

	for(auto s :mesh.GetRegion( EDGE))
	{

		auto n=mesh._C(s);

		auto expect = std::cos(Dot(k,mesh.GetCoordinates(s)))*( k[(n+1)%3]- k[(n+2)%3] )* mesh.Volume(s);

		vf1b[s]=expect;
		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] ) * mesh.Volume(s);

		variance+= abs( (vf1[s]-expect)*(vf1[s]-expect));

		average+= (vf1[s]-expect);

		if( abs(vf1[s])>1.0e-10 || abs(expect)>1.0e-10 )
		EXPECT_LE(abs(2.0*(vf1[s]-expect)/(vf1[s] + expect)), error ) << vf1[s]<<" "<<expect;

	}

	variance /= vf1.size();
	average /= vf1.size();
	CHECK(variance);
	CHECK(average);
	LOGGER<<DUMP(vf1);
	LOGGER<<DUMP(vf1b);
	LOGGER<<DUMP(vf2);
	LOGGER<<DUMP(vf2b);

}
}

TYPED_TEST_P(TestDiffCalculus, curl_grad_f0_eq_0){
{	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;
	typename TestFixture::TZeroForm f0(mesh );

	typename TestFixture::TOneForm f1(mesh );
	typename TestFixture::TOneForm f1b(mesh );
	typename TestFixture::TTwoForm f2(mesh );
	typename TestFixture::TTwoForm f2b(mesh );

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;
	f0.Clear();
	for(auto s:mesh.GetRegion(VERTEX))
	{
		f0[s]=TestFixture::default_value* uniform_dist(gen);
		m+=abs(f0[s]);
	}

	m/=f0.size();

	LOG_CMD(f1 = Grad(f0));
	LOG_CMD(f2 = Curl(f1));
	LOG_CMD(f2b = Curl(Grad(f0)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRegion(FACE))
	{
		relative_error+=abs(f2[s]);

		if(abs(f2[s])>1.0e-10)
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
		relative_error+=abs(f2b[s]);

		if(abs(f2b[s])>1.0e-10)
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

TYPED_TEST_P(TestDiffCalculus, curl_grad_f3_eq_0){
{
	auto const & mesh= TestFixture::mesh;

	typename TestFixture::TThreeForm f3 (mesh );

	typename TestFixture::TOneForm f1(mesh );
	typename TestFixture::TOneForm f1b(mesh );
	typename TestFixture::TTwoForm f2(mesh );
	typename TestFixture::TTwoForm f2b(mesh );

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;
	f3.Clear();

	for(auto s:mesh.GetRegion(VOLUME))
	{	f3[s]= uniform_dist(gen); m+=abs(f3[s]);}

	m/=f3.size();

	LOG_CMD(f2 = Grad(f3));
	LOG_CMD(f1 = Curl(f2));
	LOG_CMD(f1b = Curl(Grad(f3)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRegion(EDGE))
	{
		relative_error+=abs(f1[s]);

		if(abs(f1[s])>1.0e-10)
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

		relative_error+=abs(f1b[s]);

		if(abs(f1b[s])>1.0e-10)
		{
			++count;
		}
	}

	relative_error=relative_error/m;

	INFORM2(relative_error);

	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_EQ(0,count)<< " number of non-zero points =" << count;

}
}
TYPED_TEST_P(TestDiffCalculus, div_curl_f1_eq0){
{

	auto const & mesh= TestFixture::mesh;

	typename TestFixture::TOneForm f1(mesh );
	typename TestFixture::TOneForm f1b(mesh );
	typename TestFixture::TTwoForm f2(mesh );
	typename TestFixture::TTwoForm f2b(mesh );
	typename TestFixture::TZeroForm f0a (mesh );
	typename TestFixture::TZeroForm f0b (mesh );

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f2.Clear();

	Real m=0.0;

	for(auto s:mesh.GetRegion(FACE))
	{
		f2[s]=TestFixture::default_value* uniform_dist(gen);
		m+=abs(f2[s]);
	}

	m/=f2.size();
	LOG_CMD(f1 = Curl(f2));

	LOG_CMD(f0a = Diverge( f1));

	LOG_CMD(f0b = Diverge( Curl(f2)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRegion(VERTEX))
	{
		relative_error+=abs(f0a[s]);

		if(abs(f0a[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< " number of non-zero points =" << count;

	count =0;
	relative_error=0.0;
	for(auto s:mesh.GetRegion(VERTEX))
	{
		relative_error+=abs(f0b[s]);
		if(abs(f0b[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< " number of non-zero points =" << count;

}
}

TYPED_TEST_P(TestDiffCalculus, div_curl_f2_eq0){
{
	auto const & mesh= TestFixture::mesh;

	typename TestFixture::TOneForm f1(mesh);
	typename TestFixture::TTwoForm f2(mesh);
	typename TestFixture::TThreeForm f3a(mesh);
	typename TestFixture::TThreeForm f3b(mesh);

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f1.Clear();

	Real m=0.0;

	for(auto s:mesh.GetRegion(EDGE))
	{
		f1[s]=TestFixture::default_value* uniform_dist(gen);
		m+=abs(f1[s]);
	}

	m/=f1.size();

	LOG_CMD(f2 = Curl(f1));

	LOG_CMD(f3a = Diverge( f2));

	LOG_CMD(f3b = Diverge( Curl(f1)));

	size_t count=0;

	Real relative_error=0;

	for(auto s:mesh.GetRegion(VOLUME))
	{
		relative_error+=abs( f3a[s]);

		if(abs(f3a[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< " number of non-zero points =" << count;

	count =0;
	relative_error=0.0;
	for(auto s:mesh.GetRegion(VOLUME))
	{
		relative_error+=abs(f3b[s]);
		if(abs(f3b[s])>1.0e-10*m)
		{
			++count;
		}
	}

	relative_error=relative_error/m;
	INFORM2(relative_error);
	EXPECT_GT(1.0e-8,relative_error);
	ASSERT_DOUBLE_EQ(0,count)<< " number of non-zero points =" << count;

}
}

REGISTER_TYPED_TEST_CASE_P(TestDiffCalculus, grad, diverge, curl1, curl2, curl_grad_f0_eq_0, curl_grad_f3_eq_0,
        div_curl_f1_eq0, div_curl_f2_eq0);

#endif /* FETL_TEST3_H_ */
