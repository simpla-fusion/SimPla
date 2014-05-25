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
#include <limits>

#include "fetl.h"
#include "save_field.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;
static constexpr auto epsilon = 1e7 * std::numeric_limits<Real>::epsilon();

template<typename TParam>
class TestDiffCalculus: public testing::Test
{

protected:
	virtual void SetUp()
	{
		TParam::SetUpMesh(&mesh);
		TParam::SetDefaultValue(&default_value);
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;

	mesh_type mesh;

	static constexpr double PI = 3.141592653589793;

	static constexpr nTuple<3, Real> k =
	{ 2.0 * PI, 2.0 * PI, 4.0 * PI }; // @NOTE must   k = n TWOPI, period condition

	value_type default_value;

};
TYPED_TEST_CASE_P(TestDiffCalculus);

TYPED_TEST_P(TestDiffCalculus, grad0){
{

	typedef typename TestFixture::iterator iterator;
	typedef typename TestFixture::value_type value_type;
	auto const & mesh= TestFixture::mesh;

	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	auto kdx=Dot(k,mesh.GetDx());
	auto kdx2=kdx*kdx;

	typename TestFixture::TOneForm f1(mesh);
	typename TestFixture::TOneForm f1b(mesh);
	typename TestFixture::TZeroForm f0(mesh);

	f0.Clear();
	f1.Clear();
	f1b.Clear();
	for(auto s :mesh.GetRange( VERTEX))
	{
		f0[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	LOG_CMD(f1 = Grad(f0));

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average*= 0.0;

	for(auto s :mesh.GetRange( EDGE))
	{

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*k[mesh.ComponentNum(s.self_)];
		f1b[s]=expect;
		auto error = 0.5*kdx2;

		variance+= abs( (f1[s]-expect)*(f1[s]-expect));

		average+=(f1[s]-expect);

		if(abs(f1[s])>epsilon|| abs(expect)>epsilon)
		EXPECT_LE(abs(2.0*(f1[s]-expect)/(f1[s] + expect)), error ) << expect/f1[s]<<" "<< f1[s]<<" "<<expect;

	}

	variance /= f1.size();
	average /= f1.size();
	CHECK(variance);
	CHECK(average);
}
}

TYPED_TEST_P(TestDiffCalculus, grad3){
{

	typedef typename TestFixture::iterator iterator;
	typedef typename TestFixture::value_type value_type;
	auto const & mesh= TestFixture::mesh;

	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	auto kdx=Dot(k,mesh.GetDx());
	auto kdx2=kdx*kdx;

	typename TestFixture::TTwoForm f2(mesh);
	typename TestFixture::TTwoForm f2b(mesh);
	typename TestFixture::TThreeForm f3(mesh);

	f3.Clear();
	f2.Clear();
	f2b.Clear();

	for(auto s :mesh.GetRange( VOLUME))
	{
		f3[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	LOG_CMD(f2 = Grad(f3));

	Real m = 0.0;
	Real variance = 0;
	value_type average;
	average*= 0.0;

	for(auto s :mesh.GetRange( FACE))
	{

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*k[mesh.ComponentNum(s.self_)];
		f2b[s]=expect;
		auto error = 0.5*kdx2;

		variance+= abs( (f2[s]-expect)*(f2[s]-expect));

		average+=(f2[s]-expect);

		if(abs(f2[s])>epsilon|| abs(expect)>epsilon)
		EXPECT_LE(abs(2.0*(f2[s]-expect)/(f2[s] + expect)), error ) << expect/f2[s]<<" "<< f2[s]<<" "<< f2b[s];

	}

	variance /= f2.size();
	average /= f2.size();
	CHECK(variance);
	CHECK(average);

}
}
TYPED_TEST_P(TestDiffCalculus, diverge1){
{
	typedef typename TestFixture::iterator iterator;
	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;

	auto dx = mesh.GetDx();
	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	typename TestFixture::TOneForm f1(mesh);
	typename TestFixture::TZeroForm f0(mesh);

	f0.Clear();
	f1.Clear();

	for(auto s :mesh.GetRange( EDGE))
	{
		f1[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	f0 = Diverge(f1);

	Real variance = 0;
	value_type average = 0.0;

	for(auto s :mesh.GetRange( VERTEX))
	{

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*(k[0]+k[1]+k[2]);

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] ) * dx[0]*dx[0];

		variance+= abs( (f0[s]-expect)*(f0[s]-expect));

		average+= (f0[s]-expect);

		auto x=mesh.GetCoordinates(s);
		if((abs(f0[s])>epsilon|| abs(expect)>epsilon) && std::abs(x[2])>0.0)
		EXPECT_LE(abs(2.0*(f0[s]-expect)/(f0[s] + expect)), error )<< expect/f0[s]<<" "<< f0[s]<<" "<<expect<<" "<< (mesh.GetCoordinates(s));;

	}

	variance /= f0.size();
	average /= f0.size();
	CHECK(variance);
	CHECK(average);

}
}

TYPED_TEST_P(TestDiffCalculus, diverge2){
{
	typedef typename TestFixture::iterator iterator;
	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;

	auto dx = mesh.GetDx();
	auto k=TestFixture::k;
	auto k2 = Dot(k, k);

	typename TestFixture::TTwoForm f2(mesh);
	typename TestFixture::TThreeForm f3(mesh);

	f3.Clear();
	f2.Clear();

	for(auto s :mesh.GetRange( FACE))
	{
		f2[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	f3 = Diverge(f2);

	Real variance = 0;
	value_type average = 0.0;

	for(auto s :mesh.GetRange( VOLUME))
	{

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*(k[0]+k[1]+k[2]);

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] ) * dx[0]*dx[0];

		variance+= abs( (f3[s]-expect)*(f3[s]-expect));

		average+= (f3[s]-expect);

		if(abs(f3[s])>epsilon|| abs(expect)>epsilon)
		EXPECT_LE(abs(2.0*(f3[s]-expect)/(f3[s] + expect)), error )
		<<" "<< expect/f3[s]<<" "<< f3[s]<<" "<<expect;

	}

	variance /= f3.size();
	average /= f3.size();
	CHECK(variance);
	CHECK(average);

}
}
TYPED_TEST_P(TestDiffCalculus, curl1){
{
	typedef typename TestFixture::iterator iterator;
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

	for(auto s :mesh.GetRange(EDGE))
	{
		vf1[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf2 = Curl(vf1));

	for(auto s :mesh.GetRange( FACE))
	{
		auto n=mesh.ComponentNum(s.self_);

		auto expect= std::cos(Dot(k,mesh.GetCoordinates(s)))*( k[(n+1)%3]- k[(n+2)%3] );

		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] );

		variance+= abs( (vf2[s]-expect)*(vf2[s]-expect));

		average+= (vf2[s]-expect);
		auto x=mesh.GetCoordinates(s);
		if((abs(vf2[s])>epsilon|| abs(expect)>epsilon) && std::abs(x[2])>0.0)
		EXPECT_LE(abs(2.0*(vf2[s]-expect)/(vf2[s] + expect)), error ) << vf2[s]<<" "<<expect<<" "<<x;

	}

	variance /= vf2.size();
	average /= vf2.size();
	CHECK(variance);
	CHECK(average);

}
}

TYPED_TEST_P(TestDiffCalculus, curl2){
{
	typedef typename TestFixture::iterator iterator;
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

	for(auto s :mesh.GetRange(FACE))
	{
		vf2[s]= std::sin(Dot(k,mesh.GetCoordinates(s)));
	};

	LOG_CMD(vf1 = Curl(vf2));

	vf1b.Clear();

	for(auto s :mesh.GetRange( EDGE))
	{

		auto n=mesh.ComponentNum(s.self_);

		auto expect = std::cos(Dot(k,mesh.GetCoordinates(s)))*( k[(n+1)%3]- k[(n+2)%3] );

		vf1b[s]=expect;
		auto error = 0.5*(k[0] * k[0]+k[1] * k[1]+k[2] * k[2] );

		variance+= abs( (vf1[s]-expect)*(vf1[s]-expect));

		average+= (vf1[s]-expect);

		auto x=mesh.GetCoordinates(s);
		if((abs(vf1[s])>epsilon|| abs(expect)>epsilon) && std::abs(x[2])>0.0)
		EXPECT_LE(abs(2.0*(vf1[s]-expect)/(vf1[s] + expect)), error ) << vf1[s]<<" "<<expect<<" "<<mesh.GetCoordinates(s);

	}

	variance /= vf1.size();
	average /= vf1.size();
	CHECK(variance);
	CHECK(average);

}
}

TYPED_TEST_P(TestDiffCalculus, identity_curl_grad_f0_eq_0){
{	typedef typename TestFixture::value_type value_type;

	auto const & mesh= TestFixture::mesh;
	typename TestFixture::TZeroForm f0(mesh );

	typename TestFixture::TOneForm f1(mesh );
	typename TestFixture::TTwoForm f2a(mesh );
	typename TestFixture::TTwoForm f2b(mesh );

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;
	f0.Clear();
	for(auto s:mesh.GetRange(VERTEX))
	{

		auto a= uniform_dist(gen);
		f0[s]=TestFixture::default_value* a;
		m+=a*a;
	}

	m =std::sqrt(m)*abs(TestFixture::default_value);

	LOG_CMD(f1 = Grad(f0));
	LOG_CMD(f2a = Curl(f1));
	LOG_CMD(f2b = Curl(Grad(f0)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRange(FACE))
	{

		relative_error+=abs(f2b[s]);
		EXPECT_EQ( (f2a[s]), (f2b[s]));
	}

	relative_error/=m;

	INFORM2(relative_error);
	EXPECT_LE(relative_error,epsilon);

}
}

TYPED_TEST_P(TestDiffCalculus, identity_curl_grad_f3_eq_0){
{
	auto const & mesh= TestFixture::mesh;

	typename TestFixture::TThreeForm f3 (mesh );
	typename TestFixture::TOneForm f1a(mesh );
	typename TestFixture::TOneForm f1b(mesh );
	typename TestFixture::TTwoForm f2(mesh );
	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	Real m=0.0;

	f3.Clear();

	for(auto s:mesh.GetRange(VOLUME))
	{
		auto a= uniform_dist(gen);
		f3[s]=a*TestFixture::default_value;
		m+=a*a;
	}

	m =std::sqrt(m)*abs(TestFixture::default_value);

	LOG_CMD(f2 = Grad(f3));
	LOG_CMD(f1a = Curl(f2));
	LOG_CMD(f1b = Curl(Grad(f3)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRange(EDGE))
	{

		EXPECT_EQ( (f1a[s]), (f1b[s]));

		relative_error+=abs(f1b[s]);

	}

	relative_error /=m;

	INFORM2(relative_error);
	EXPECT_LE(relative_error,epsilon);

}
}
TYPED_TEST_P(TestDiffCalculus, identity_div_curl_f1_eq0){
{

	auto const & mesh= TestFixture::mesh;

	typename TestFixture::TOneForm f1(mesh );
	typename TestFixture::TTwoForm f2(mesh );
	typename TestFixture::TZeroForm f0a (mesh );
	typename TestFixture::TZeroForm f0b (mesh );

	std::mt19937 gen;
	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	f2.Clear();

	Real m=0.0;

	for(auto s:mesh.GetRange(FACE))
	{
		auto a= uniform_dist(gen);

		f2[s]=TestFixture::default_value* uniform_dist(gen);

		m+=a*a;
	}

	m =std::sqrt(m)*abs(TestFixture::default_value);

	LOG_CMD(f1 = Curl(f2));

	LOG_CMD(f0a = Diverge( f1));

	LOG_CMD(f0b = Diverge( Curl(f2)));

	size_t count=0;
	Real relative_error=0;

	for(auto s:mesh.GetRange(VERTEX))
	{
		relative_error+=abs(f0b[s]);
		EXPECT_EQ( (f0a[s]), (f0b[s]));
	}

	relative_error/=m;
	INFORM2(relative_error);
	EXPECT_LE(relative_error,epsilon);

}
}

TYPED_TEST_P(TestDiffCalculus, identity_div_curl_f2_eq0){
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

	for(auto s:mesh.GetRange(EDGE))
	{
		auto a= uniform_dist(gen);
		f1[s]=TestFixture::default_value*a;
		m+=a*a;
	}

	m =std::sqrt(m)*abs(TestFixture::default_value);

	LOG_CMD(f2 = Curl(f1));

	LOG_CMD(f3a = Diverge( f2));

	LOG_CMD(f3b = Diverge( Curl(f1)));

	size_t count=0;

	Real relative_error=0;

	for(auto s:mesh.GetRange(VOLUME))
	{

		EXPECT_DOUBLE_EQ(abs(f3a[s]),abs(f3b[s]));

		relative_error+=abs(f3b[s]);

	}

	relative_error/= m;
	INFORM2(relative_error);
	EXPECT_LE(relative_error,epsilon);

}
}

REGISTER_TYPED_TEST_CASE_P(TestDiffCalculus, grad0, grad3, diverge1, diverge2, curl1, curl2, identity_curl_grad_f0_eq_0,
		identity_curl_grad_f3_eq_0, identity_div_curl_f1_eq0, identity_div_curl_f2_eq0);

#endif /* FETL_TEST3_H_ */
