/*
 * array_test.cpp
 *
 *  Created on: 2012-1-12
 *      Author: salmon
 */

#include "array.h"

#include <gtest/gtest.h>
#include <iostream>
#include <complex>
#include "primitives/ntuple.h"
#include "primitives/expression.h"

using namespace simpla;

template<typename T> T _real(T const & v)
{
	return (v);
}

template<typename T> T _real(std::complex<T> const & v)
{
	return (real(v));
}
#define EQUATION(_A,_B,_C,_a,_b,_c)  ( (- _A /_a - _a )* ( _B * _b  - _c ) - _C )

template<typename T>
class TestArray: public testing::Test
{
protected:

	virtual void SetUp()
	{
//		a = static_cast<Value>(1);
//		b = static_cast<Value>(3);
//		c = static_cast<Value>(4);
//		d = static_cast<Value>(7);
//		m = static_cast<Value>(loop_);
		a = 1;
		b = 2;
		c = 3;
		d = 4;

		for (size_t i = 0; i < num; ++i)
		{
			aA[i] = i * 2;
			aB[i] = 5 - i;
			aC[i] = i * 5 + 1;
			aD[i] = EQUATION(aA[i],aB[i],aC[i],a,b,c);
		}

		T(num).swap(vA);
		T(num).swap(vB);
		T(num).swap(vC);
		T(num).swap(vD);
	}

public:
	static const int num = 100000;
	static const int loop_ = 1000000L;
	typedef typename T::Value Value;
	Value m;

	T vA, vB, vC, vD;
	Value a, b, c, d;
	Value aA[num], aB[num], aC[num], aD[num];
};

typedef testing::Types< //
		Array<double> //
		, Array<std::complex<double> > //
> MyTypes;

TYPED_TEST_CASE(TestArray, MyTypes);

TYPED_TEST(TestArray,Assign_Scalar){
{

	TestFixture::vD = TestFixture::a;

	for (size_t i = 0; i < TestFixture::num; ++i)
	{
		EXPECT_DOUBLE_EQ(_real(TestFixture::a ), _real(TestFixture::vD[i]) );
	}
}}

TYPED_TEST(TestArray,Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

#pragma omp parallel for
	for (size_t i = 0; i < TestFixture::num; ++i)
	{
		EXPECT_DOUBLE_EQ( _real(TestFixture::aA[i]), _real(TestFixture::vA[i]));
	}
}}
//
////TYPED_TEST(TestArray, Logical){
////{
////	EXPECT_TRUE( TestFixture::vA!= TestFixture::vB);
////	EXPECT_TRUE( TestFixture::vA== TestFixture::vA);
////	EXPECT_FALSE( TestFixture::vA== TestFixture::vB);
////
////}}
//

TYPED_TEST(TestArray, Function){
{
	TestFixture::vD = sin(TestFixture::vB);

	for (size_t i = 0; i < TestFixture::num; ++i)
	{
		EXPECT_DOUBLE_EQ(_real(TestFixture::vD[i]),_real( std::sin(TestFixture::vB[i])));
	}
}
}

TYPED_TEST(TestArray, Arithmetic){
{
	TestFixture::vA= TestFixture::aA;
	TestFixture::vB= TestFixture::aB;
	TestFixture::vC= TestFixture::aC;

	TestFixture::vD = EQUATION( TestFixture::vA , TestFixture::vB , TestFixture::vC,
			TestFixture::a, TestFixture::b, TestFixture::c);

	for (size_t i = 0; i < TestFixture::num; ++i)
	{
		EXPECT_DOUBLE_EQ(_real(EQUATION(TestFixture::vA[i] ,TestFixture::vB[i] ,TestFixture::vC[i],
								TestFixture::a, TestFixture::b, TestFixture::c)),_real( TestFixture::vD[i]));
		EXPECT_DOUBLE_EQ(_real(TestFixture::vD[i]),_real( TestFixture::aD[i]));
	}
}
}

template<typename T>
class TestVecArray: public testing::Test
{
protected:

	virtual void SetUp()
	{
//		a = static_cast<Value>(1);
//		b = static_cast<Value>(3);
//		c = static_cast<Value>(4);
//		d = static_cast<Value>(7);
//		m = static_cast<Value>(loop_);
		a = 1;
		b = 2;
		vc = 3;
		vd = 4;

		for (size_t i = 0; i < num; ++i)
		{
			aA[i] = i * 2 * vc;
			aB[i] = (5 - i) * vd;
			aC[i] = (i * 5 + 1) * vc;
			aD[i] = EQUATION(aA[i],aB[i],aC[i],a,b,b);
		}

		T(num).swap(vA);
		T(num).swap(vB);
		T(num).swap(vC);
		T(num).swap(vD);
	}

public:
	static const int num = 100000;
	static const int loop_ = 1000000L;
	typedef typename T::Value Value;
	Value m;

	T vA, vB, vC, vD;
	Real a, b;
	Value vc, vd;
	Value aA[num], aB[num], aC[num], aD[num];
};

typedef testing::Types< //
		Array<nTuple<3, double> > //
		, Array<nTuple<5, std::complex<double> > > //
> MyVecTypes;

TYPED_TEST_CASE(TestVecArray, MyVecTypes);

TYPED_TEST(TestVecArray,Assign_Scalar){
{

	TestFixture::vA = TestFixture::vc;

	for (size_t i = 0; i < TestFixture::num; ++i)
	{
		typename TestFixture::Value res;res=(TestFixture::vc -TestFixture::vA[i]);

		EXPECT_DOUBLE_EQ(abs(res) ,0)<<TestFixture::vc <<TestFixture::vA[i];
	}
}}

TYPED_TEST(TestVecArray,Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

#pragma omp parallel for
	for (size_t i = 0; i < TestFixture::num; ++i)
	{
		typename TestFixture::Value res;res=(TestFixture::aA[i]-TestFixture::vA[i]);
		EXPECT_DOUBLE_EQ(abs(res) ,0.0)<<res;
	}
}}

//
//TYPED_TEST(TestArray, performance_rawarray){
//{
//
//	for (int s = 0; s < TestFixture::loop_; ++s)
//	{
//		for(int i=0;i<TestFixture::num;++i)
//		{	TestFixture::aD[i] += EQUATION(aA[i] ,aB[i] ,aC[i]);};
//	}
//
////	for (int i = 0; i < TestFixture::num; ++i)
////	{
////		EXPECT_DOUBLE_EQ(_real(EQUATION(aA[i] ,aB[i] ,aC[i])),_real(TestFixture::aD[i]/TestFixture::m) );
////	}
//
//}
//}
//TYPED_TEST(TestArray, performance_nTuple){
//{
//
//	for (int s = 0; s < TestFixture::loop_; ++s)
//	{
//		TestFixture::vD += EQUATION(vA ,vB ,vC);
//	}
//
////	for (int i = 0; i < TestFixture::num; ++i)
////	{
////		EXPECT_DOUBLE_EQ(_real(TestFixture::aD[i]) ,_real(TestFixture::vD[i]/TestFixture::m));
////	}
//}
//}
