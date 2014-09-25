/*
 * test_ntuple.cpp
 *
 *  created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ntuple.h"
#include "ntuple_et.h"
#include "log.h"
#include "pretty_stream.h"

using namespace simpla;

#define EQUATION(_A,_B,_C)  ( -(TestFixture::_A  +TestFixture::a )/(   TestFixture::_B *TestFixture::b -TestFixture::c  )- TestFixture::_C)

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
		for (int i = 0; i < DIMENSION; ++i)
		{
			aA[i] = i * 2;
			aB[i] = 5 - i;
			aC[i] = i * 5 + 1;
			aD[i] = 0;
			vA[i] = aA[i];
			vB[i] = aB[i];
			vC[i] = aC[i];
			vD[i] = 0;

			res[i] = -(aA[i] + a) / (aB[i] * b - c) - aC[i];

		}

		m = 1000000L;
	}

public:
	static constexpr unsigned int DIMENSION = T::DIMENSION;
	typedef typename T::value_type value_type;

	size_t m = 10000000L;

	T vA, vB, vC, vD;
	value_type aA[DIMENSION], aB[DIMENSION], aC[DIMENSION], aD[DIMENSION],
			res[DIMENSION];
	value_type a, b, c, d;

};

typedef testing::Types<

nTuple<3, double>

, nTuple<10, double>, nTuple<20, double>

, nTuple<10, std::complex<double> >

, nTuple<3, std::complex<double> >, nTuple<20, std::complex<double> >

, nTuple<3, int>, nTuple<10, int>, nTuple<20, int>

> nTupleTypes;

TYPED_TEST_CASE(TestNtuple, nTupleTypes);

TYPED_TEST(TestNtuple,Assign_Scalar){
{

	TestFixture::vD = TestFixture::aA;

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	{
		EXPECT_DOUBLE_EQ(abs(TestFixture::aA[i]), abs(TestFixture::vD[i]) );
	}
}}

TYPED_TEST(TestNtuple,Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	{
		EXPECT_DOUBLE_EQ( abs(TestFixture::aA[i]), abs(TestFixture::vA[i]));
	}
}}

TYPED_TEST(TestNtuple, Arithmetic){
{
	TestFixture::vD = EQUATION(vA ,vB ,vC);

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	{
		EXPECT_DOUBLE_EQ(abs(EQUATION(vA[i] ,vB[i] ,vC[i])),abs( TestFixture::vD[i]));
	}
}
}

TYPED_TEST(TestNtuple, self_assign){
{
//	CHECK(TestFixture::vB );
//	CHECK(TestFixture::vA );
	TestFixture::vB +=TestFixture::vA;
//	CHECK(TestFixture::vB );
}
}
TYPED_TEST(TestNtuple, Dot){
{
	typename TestFixture::value_type res;

	res=0;

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	{
		res += TestFixture::vA[i] * TestFixture::vB[i];
	}
	EXPECT_DOUBLE_EQ(abs(res),abs( dot( TestFixture::vA, TestFixture::vB)));
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

	vC=cross(vA,vB);

	EXPECT_EQ(vD ,vC);
}}
//
TYPED_TEST(TestNtuple, performance_rawarray){
{
	for (size_t s = 0; s < TestFixture::m; ++s)
	{
		for(int i=0;i<TestFixture::DIMENSION;++i)
		{	TestFixture::aD[i] +=EQUATION(aA[i] ,aB[i] ,aC[i])*s;};
	}

//	for (int i = 0; i < TestFixture::DIMENSION; ++i)
//	{
//		EXPECT_DOUBLE_EQ(abs(EQUATION(aA[i] ,aB[i] ,aC[i])),abs(TestFixture::aD[i]/TestFixture::m) );
//	}

}
}
TYPED_TEST(TestNtuple, performance_nTuple){
{

	for (size_t s = 0; s < TestFixture::m; ++s)
	{
		TestFixture::vD +=EQUATION(vA ,vB ,vC)*(s);
	}

//	for (int i = 0; i < TestFixture::DIMENSION; ++i)
//	{
//		EXPECT_DOUBLE_EQ(abs(EQUATION(vA[i] ,vB[i] ,vC[i])) ,abs(TestFixture::vD[i]/TestFixture::m));
//	}
}
}

template<typename T>
class TestNMatrix: public testing::Test
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
		for (int i = 0; i < DIMENSION; ++i)
			for (int j = 0; j < MDIMS; ++j)
			{
				aA[i][j] = i * 2;
				aB[i][j] = 5 - i;
				aC[i][j] = i * 5 + 1;
				aD[i][j] = (aA[i][j] + a) / (aB[i][j] * b + c) - aC[i][j];
				vA[i][j] = aA[i][j];
				vB[i][j] = aB[i][j];
				vC[i][j] = aC[i][j];
				vD[i][j] = 0;
			}

		m = 1000000L;
	}

public:
	static constexpr unsigned int DIMENSION = T::DIMENSION;
	static constexpr unsigned int MDIMS = T::value_type::DIMENSION;

	typedef typename T::value_type::value_type value_type;

	value_type m;

	T vA, vB, vC, vD;
	value_type aA[DIMENSION][MDIMS], aB[DIMENSION][MDIMS], aC[DIMENSION][MDIMS],
			aD[DIMENSION][MDIMS];
	value_type a, b, c, d;

};

typedef testing::Types<

nTuple<3, nTuple<3, double>>,

nTuple<3, nTuple<3, Complex>>,

nTuple<3, nTuple<3, int>>,

nTuple<10, nTuple<5, double>>,

nTuple<10, nTuple<10, double>>

> MatrixTypes;

TYPED_TEST_CASE(TestNMatrix, MatrixTypes);

TYPED_TEST(TestNMatrix,Assign_Scalar){
{

	TestFixture::vD = TestFixture::aA;

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	for (int j = 0; j < TestFixture::MDIMS; ++j)

	{
		EXPECT_DOUBLE_EQ(abs(TestFixture::aA[i][j]), abs(TestFixture::vD[i][j]) );
	}
}}

TYPED_TEST(TestNMatrix,Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	for (int j = 0; j < TestFixture::MDIMS; ++j)
	{
		EXPECT_DOUBLE_EQ( abs(TestFixture::aA[i][j]), abs(TestFixture::vA[i][j]));
	}
}}

TYPED_TEST(TestNMatrix, Arithmetic){
{
	TestFixture::vD = EQUATION(vA ,vB ,vC);

	for (int i = 0; i < TestFixture::DIMENSION; ++i)
	for (int j = 0; j < TestFixture::MDIMS; ++j)
	{
		EXPECT_DOUBLE_EQ(abs(EQUATION(vA[i][j] ,vB[i][j] ,vC[i][j])),abs( TestFixture::vD[i][j]));
	}
}
}

TYPED_TEST(TestNMatrix, performance_rawarray){
{

	for (size_t s = 0; s < 10000000L; ++s)
	{
		for(int i=0;i<TestFixture::DIMENSION;++i)
		for (int j = 0; j < TestFixture::MDIMS; ++j)
		{	TestFixture::aD[i][j] +=EQUATION(aA[i][j] ,aB[i][j] ,aC[i][j])*s;};
	}

//	for (int i = 0; i < TestFixture::DIMENSION; ++i)
//	{
//		EXPECT_DOUBLE_EQ(abs(EQUATION(aA[i] ,aB[i] ,aC[i])),abs(TestFixture::aD[i]/TestFixture::m) );
//	}

}
}
TYPED_TEST(TestNMatrix, performance_nTuple){
{

	for (size_t s = 0; s < 10000000L; ++s)
	{
		TestFixture::vD +=EQUATION(vA ,vB ,vC)*(s);
	}

//	for (int i = 0; i < TestFixture::DIMENSION; ++i)
//	{
//		EXPECT_DOUBLE_EQ(abs(TestFixture::aD[i]) ,abs(TestFixture::vD[i]/TestFixture::m));
//	}
}
}
