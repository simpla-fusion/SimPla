/*
 * test_ntuple.cpp
 *
 *  Created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include <complex>
#include "fetl.h"
#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"

using namespace simpla;

//template<typename T> T _abs(T const & v)
//{
//	return (v);
//}
//
//template<typename T> T _real(std::complex<T> const & v)
//{
//	return (real(v));
//}
#define EQUATION(_A,_B,_C)  ( -TestFixture::_A  *TestFixture::a  + TestFixture::a * TestFixture::_A   -   TestFixture::_B /TestFixture::b   - TestFixture::_C)

template<typename T>
class TestNtuple: public testing::Test
{
protected:

	virtual void SetUp()
	{
//		a = static_cast<value_type>(1);
//		b = static_cast<value_type>(3);
//		c = static_cast<value_type>(4);
//		d = static_cast<value_type>(7);

		a = 1;
		b = 3;
		c = 4;
		d = 7;
		for (int i = 0; i < NUM_OF_DIMS; ++i)
		{
			aA[i] = i * 2;
			aB[i] = 5 - i;
			aC[i] = i * 5 + 1;
			aD[i] = (aA[i] + a) / (aB[i] * b + c) - aC[i];
			vA[i] = aA[i];
			vB[i] = aB[i];
			vC[i] = aC[i];
			vD[i] = 0;
		}

		m = 1000000L;
	}

public:
	static constexpr int NUM_OF_DIMS = T::NUM_OF_DIMS;
	typedef typename T::value_type value_type;
	value_type m;

	T vA, vB, vC, vD;
	value_type aA[NUM_OF_DIMS], aB[NUM_OF_DIMS], aC[NUM_OF_DIMS], aD[NUM_OF_DIMS];
	value_type a, b, c, d;

};

typedef testing::Types<

nTuple<3, double>, nTuple<10, double>, nTuple<20, double>

, nTuple<10, std::complex<double> >, nTuple<3, std::complex<double> >, nTuple<20, std::complex<double> >

, nTuple<3, int>, nTuple<10, int>, nTuple<20, int>

//, nTuple<3, nTuple<3, double>>

> MyTypes;

TYPED_TEST_CASE(TestNtuple, MyTypes);

TYPED_TEST(TestNtuple,Assign_Scalar){
{

	TestFixture::vD = TestFixture::aA;

	for (int i = 0; i < TestFixture::NUM_OF_DIMS; ++i)
	{
		EXPECT_DOUBLE_EQ(abs(TestFixture::aA[i]), abs(TestFixture::vD[i]) );
	}
}}

TYPED_TEST(TestNtuple,Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

	for (int i = 0; i < TestFixture::NUM_OF_DIMS; ++i)
	{
		EXPECT_DOUBLE_EQ( abs(TestFixture::aA[i]), abs(TestFixture::vA[i]));
	}
}}

//TYPED_TEST(TestNtuple, Logical){
//{
//	EXPECT_TRUE( TestFixture::vA!= TestFixture::vB);
//	EXPECT_TRUE( TestFixture::vA== TestFixture::vA);
//	EXPECT_FALSE( TestFixture::vA== TestFixture::vB);
//
//}}
//
TYPED_TEST(TestNtuple, Arithmetic){
{
	TestFixture::vD = EQUATION(vA ,vB ,vC);

	for (int i = 0; i < TestFixture::NUM_OF_DIMS; ++i)
	{
		EXPECT_DOUBLE_EQ(abs(EQUATION(vA[i] ,vB[i] ,vC[i])),abs( TestFixture::vD[i]));
	}
}
}

TYPED_TEST(TestNtuple, Dot){
{
	typename TestFixture::value_type res;

	res=0;

	for (int i = 0; i < TestFixture::NUM_OF_DIMS; ++i)
	{
		res += TestFixture::vA[i] * TestFixture::vB[i];
	}
	EXPECT_DOUBLE_EQ(abs(res),abs( Dot( TestFixture::vA, TestFixture::vB)));
}}

TYPED_TEST(TestNtuple, Cross){
{
	nTuple<3, typename TestFixture::value_type> vA, vB,vC ,vD;

	for (int i = 0; i < 3; ++i)
	{
		vA[i] = (i * 2);
		vB[i] = (5 - i);
	}

	for (int i = 0; i < 3; ++i)
	{
		vD[i] = vA[(i + 1) % 3] * vB[(i + 2) % 3]
		- vA[(i + 2) % 3] * vB[(i + 1) % 3];
	}

	vC=Cross(vA,vB);

	EXPECT_EQ(vD ,vC);
}}

TYPED_TEST(TestNtuple, performance_rawarray){
{

	for (size_t s = 0; s < 10000000L; ++s)
	{
		for(int i=0;i<TestFixture::NUM_OF_DIMS;++i)
		{	TestFixture::aD[i] += (s)*EQUATION(aA[i] ,aB[i] ,aC[i]);};
	}

//	for (int i = 0; i < TestFixture::NUM_OF_DIMS; ++i)
//	{
//		EXPECT_DOUBLE_EQ(abs(EQUATION(aA[i] ,aB[i] ,aC[i])),abs(TestFixture::aD[i]/TestFixture::m) );
//	}

}
}
TYPED_TEST(TestNtuple, performance_nTuple){
{

	for (size_t s = 0; s < 10000000L; ++s)
	{
		TestFixture::vD += (s)*EQUATION(vA ,vB ,vC);
	}

//	for (int i = 0; i < TestFixture::NUM_OF_DIMS; ++i)
//	{
//		EXPECT_DOUBLE_EQ(abs(TestFixture::aD[i]) ,abs(TestFixture::vD[i]/TestFixture::m));
//	}
}
}
