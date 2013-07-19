/*
 * test_ntuple.cpp
 *
 *  Created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include <complex>
#include "ntuple.h"
#include "expression_template/arithmetic.h"

using namespace simpla;

template<typename T> T _real(T const & v)
{
	return (v);
}

template<typename T> T _real(std::complex<T> const & v)
{
	return (real(v));
}
#define EQUATION(_A,_B,_C)  ( (-TestFixture::_A /TestFixture::a - TestFixture::a )* ( TestFixture::_B * TestFixture::b  - TestFixture::b ) - TestFixture::_C)

template<typename T>
class TestNtuple: public testing::Test
{
protected:

	virtual void SetUp()
	{
		a = static_cast<Value>(1);
		b = static_cast<Value>(3);
		c = static_cast<Value>(4);
		d = static_cast<Value>(7);

		for (int i = 0; i < NDIM; ++i)
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

		m = static_cast<Value>(loop_num);
	}

public:
	static const int NDIM = T::NDIM;
	static const int loop_num = 1000000L;
	typedef typename T::Value Value;
	Value m;

	T vA, vB, vC, vD;
	Value aA[NDIM], aB[NDIM], aC[NDIM], aD[NDIM];
	Value a, b, c, d;

};

typedef testing::Types< //
		nTuple<3, double>, nTuple<3, int>
		, nTuple<3, std::complex<double> >
		, nTuple<10, double>, nTuple<20, double>

> MyTypes;

TYPED_TEST_CASE(TestNtuple, MyTypes);

TYPED_TEST(TestNtuple,Assign_Scalar){
{

	TestFixture::vD = TestFixture::aA;

	for (int i = 0; i < TestFixture::NDIM; ++i)
	{
		EXPECT_DOUBLE_EQ(_real(TestFixture::aA[i]), _real(TestFixture::vD[i]) );
	}
}}

TYPED_TEST(TestNtuple,Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

	for (int i = 0; i < TestFixture::NDIM; ++i)
	{
		EXPECT_DOUBLE_EQ( _real(TestFixture::aA[i]), _real(TestFixture::vA[i]));
	}
}}

TYPED_TEST(TestNtuple, Logical){
{
	EXPECT_TRUE( TestFixture::vA!= TestFixture::vB);
	EXPECT_TRUE( TestFixture::vA== TestFixture::vA);
	EXPECT_FALSE( TestFixture::vA== TestFixture::vB);

}}

TYPED_TEST(TestNtuple, Arithmetic){
{
	TestFixture::vD = EQUATION(vA ,vB ,vC);

	for (int i = 0; i < TestFixture::NDIM; ++i)
	{
		EXPECT_DOUBLE_EQ(_real(EQUATION(vA[i] ,vB[i] ,vC[i])),_real( TestFixture::vD[i]));
	}
}
}

TYPED_TEST(TestNtuple, Dot){
{
	typename TestFixture::Value res(0);

	for (int i = 0; i < TestFixture::NDIM; ++i)
	{
		res += TestFixture::vA[i] * TestFixture::vB[i];
	}
	EXPECT_DOUBLE_EQ(_real(res),_real( Dot( TestFixture::vA, TestFixture::vB)));
}}

TYPED_TEST(TestNtuple, Cross){
{
	nTuple<3, typename TestFixture::Value> vA, vB, vD;

	for (int i = 0; i < 3; ++i)
	{
		vA[i] = static_cast<typename TestFixture::Value>(i * 2);
		vB[i] = static_cast<typename TestFixture::Value>(5 - i);
	}

	for (int i = 0; i < 3; ++i)
	{
		vD[i] = vA[(i + 1) % 3] * vB[(i + 2) % 3]
		- vA[(i + 2) % 3] * vB[(i + 1) % 3];
	}

	EXPECT_EQ(vD,Cross(vA,vB) );
}}

TYPED_TEST(TestNtuple, performance_rawarray){
{

	for (int s = 0; s < TestFixture::loop_num; ++s)
	{
		for(int i=0;i<TestFixture::NDIM;++i)
		{	TestFixture::aD[i] += EQUATION(aA[i] ,aB[i] ,aC[i]);};
	}

//	for (int i = 0; i < TestFixture::NDIM; ++i)
//	{
//		EXPECT_DOUBLE_EQ(_real(EQUATION(aA[i] ,aB[i] ,aC[i])),_real(TestFixture::aD[i]/TestFixture::m) );
//	}

}
}
TYPED_TEST(TestNtuple, performance_nTuple){
{

	for (int s = 0; s < TestFixture::loop_num; ++s)
	{
		TestFixture::vD += EQUATION(vA ,vB ,vC);
	}

//	for (int i = 0; i < TestFixture::NDIM; ++i)
//	{
//		EXPECT_DOUBLE_EQ(_real(TestFixture::aD[i]) ,_real(TestFixture::vD[i]/TestFixture::m));
//	}
}
}
