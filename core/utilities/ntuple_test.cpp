/*
 * testnTuple.cpp
 *
 *  created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>
//#include "expression_template.h"
#include "sp_type_traits.h"
#include "sp_integer_sequence.h"
#include "ntuple.h"
//#include "log.h"
//#include "pretty_stream.h"

using namespace simpla;

#define EQUATION(_A,_B,_C)  ( -(_A  +TestFixture::a )/(   _B *TestFixture::b -TestFixture::c  )- _C)

template<typename T>
class TestNtuple: public testing::Test
{
protected:

	virtual void SetUp()
	{

		a = 1;
		b = 3;
		c = 4;
		d = 7;

		DIMENSIONS = seq2ntuple(dimensions());

		seq_for_each(dimensions(),

		[&](size_t const idx[dimensions::size()])
		{
			get_value2(aA,idx) = idx[0] * 2;
			get_value2(aB,idx) = 5 - idx[0];
			get_value2(aC,idx) = idx[0] * 5 + 1;
			get_value2(aD,idx) = 0;
			get_value2(vA,idx) = get_value2(aA,idx);
			get_value2(vB,idx) = get_value2(aB,idx);
			get_value2(vC,idx) = get_value2(aC,idx);
			get_value2(vD,idx) = 0;

			get_value2(res,idx) = -(get_value2(aA,idx) + a) /
			(get_value2(aB,idx) * b - c) - get_value2(aC,idx);

		});

		num_of_loops = 1000000L;
	}

public:

	typedef T type;

	typedef typename nTuple_traits<T>::dimensions dimensions;

	nTuple<std::size_t, dimensions::size()> DIMENSIONS;

	typedef typename T::value_type value_type;

	std::size_t num_of_loops = 10000000L;

	T vA, vB, vC, vD;
	typename T::data_type aA, aB, aC, aD, res;
	value_type a, b, c, d;

};

typedef testing::Types<

nTuple<double, 3>

, nTuple<double, 10>

, nTuple<double, 20>

, nTuple<int, 3>

, nTuple<int, 10>

, nTuple<int, 20>
//
//, nTuple<std::complex<double>, 3>
//
//, nTuple<std::complex<double>, 10>
//
//, nTuple<std::complex<double>, 20>

> nTupleTypes;

TYPED_TEST_CASE(TestNtuple, nTupleTypes);

TYPED_TEST(TestNtuple, swap){
{

	simpla::swap(TestFixture::vA, TestFixture::vB);

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				EXPECT_DOUBLE_EQ(0, abs(get_value2(TestFixture::aA,idx)- get_value2(TestFixture:: vB,idx)));
				EXPECT_DOUBLE_EQ(0, abs(get_value2(TestFixture::aB,idx)- get_value2(TestFixture:: vA,idx)));
			});

}
}

TYPED_TEST(TestNtuple, reduce){
{
	typename TestFixture::value_type expect=0;

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				expect+=get_value2(TestFixture:: aA,idx);
			}
	);

	auto value=seq_reduce(typename TestFixture::dimensions(), _impl::plus(), TestFixture::vA);

	EXPECT_DOUBLE_EQ(0,abs(expect -value));

}
}

TYPED_TEST(TestNtuple, compare){
{
//	EXPECT_TRUE( TestFixture::vA==TestFixture::aA);
//	EXPECT_FALSE( TestFixture::vA==TestFixture::vB);

	EXPECT_TRUE( TestFixture::vA!=TestFixture::vB);
	EXPECT_FALSE( TestFixture::vA!=TestFixture::vA);

}
}

TYPED_TEST(TestNtuple, Assign_Scalar){
{

	TestFixture::vA = TestFixture::a;

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				EXPECT_DOUBLE_EQ(0, abs(TestFixture::a- get_value2(TestFixture:: vA,idx)));
			}
	);

}}

TYPED_TEST(TestNtuple, Assign_Array){
{
	TestFixture::vA = TestFixture::aA;

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				EXPECT_DOUBLE_EQ(0, abs(get_value2(TestFixture::aA,idx)- get_value2(TestFixture:: vA,idx)));
			}
	);

}}

TYPED_TEST(TestNtuple, Arithmetic){
{
	TestFixture::vD = EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC);

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				auto &ta=get_value2(TestFixture::vA,idx);
				auto &tb=get_value2(TestFixture::vB,idx);
				auto &tc=get_value2(TestFixture::vC,idx);
				auto &td=get_value2(TestFixture::vD,idx);

				EXPECT_DOUBLE_EQ(0, abs(EQUATION(ta,tb,tc ) - td));
			}
	);

}
}

TYPED_TEST(TestNtuple, self_assign){
{
	TestFixture::vB +=TestFixture::vA;

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				EXPECT_DOUBLE_EQ(0,abs( get_value2(TestFixture::vB,idx)
								- (get_value2(TestFixture::aB,idx)+
										get_value2(TestFixture::aA,idx))));

			}
	);

}
}

TYPED_TEST(TestNtuple, performance_rawarray){
{
	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
	{
		seq_for_each(typename TestFixture::dimensions(),
				[&](size_t const idx[TestFixture::dimensions::size()])
				{
					get_value2(TestFixture::aD,idx) +=EQUATION(get_value2(TestFixture::aA,idx),
							get_value2(TestFixture::aB,idx),
							get_value2(TestFixture::aC,idx))
					*s;

				}
		)

		;
	}

}
}
TYPED_TEST(TestNtuple, performancenTuple){
{

	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
	{
		TestFixture::vD +=EQUATION(TestFixture::vA ,TestFixture::vB ,TestFixture::vC)*(s);
	}

}
}

TYPED_TEST(TestNtuple, inner_product){
{
	typename TestFixture::value_type res;

	res=0;

	seq_for_each(typename TestFixture::dimensions(),
			[&](size_t const idx[TestFixture::dimensions::size()])
			{
				res += get_value2(TestFixture::aA,idx) * get_value2(TestFixture::aB,idx);
			}
	);

	EXPECT_DOUBLE_EQ(0,abs(res- inner_product( TestFixture::vA, TestFixture::vB)));
}}

TYPED_TEST(TestNtuple, Cross){
{
	nTuple< typename TestFixture::value_type,3> vA, vB,vC ,vD;

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

	EXPECT_DOUBLE_EQ(0,abs(vD- vC));
}}

//
//template<typename T>
//class TestNMatrix: public testing::Test
//{
//protected:
//
//	virtual void SetUp()
//	{
////		a = static_cast<value_type>(1);
////		b = static_cast<value_type>(3);
////		c = static_cast<value_type>(4);
////		d = static_cast<value_type>(7);
//
//		a = 1;
//		b = 3;
//		c = 4;
//		d = 7;
//		for (int i = 0; i < NDIMS; ++i)
//			for (int j = 0; j < MDIMS; ++j)
//			{
//				aA[i][j] = i * 2;
//				aB[i][j] = 5 - i;
//				aC[i][j] = i * 5 + 1;
//				aD[i][j] = (aA[i][j] + a) / (aB[i][j] * b + c) - aC[i][j];
//				vA[i][j] = aA[i][j];
//				vB[i][j] = aB[i][j];
//				vC[i][j] = aC[i][j];
//				vD[i][j] = 0;
//			}
//
//		m = 1000000L;
//	}
//
//public:
//	static constexpr std::size_t NDIMS = seq_get_value<0,
//			typename nTuple_traits<T>::dimensions>::value;
//
//	static constexpr std::size_t MDIMS = seq_get_value<1,
//			typename nTuple_traits<T>::dimensions>::value;
//
//	typedef typename nTuple_traits<T>::value_type value_type;
//
//	value_type m;
//
//	T vA, vB, vC, vD;
//	value_type aA[NDIMS][MDIMS], aB[NDIMS][MDIMS], aC[NDIMS][MDIMS],
//			aD[NDIMS][MDIMS];
//	value_type a, b, c, d;
//
//};
//
//typedef testing::Types<
//
//Matrix<double, 3, 3>
////
////, Matrix<Complex, 3, 3>
////
////, Matrix<int, 5, 10>
////
////, Matrix<double, 10, 5>
////
////, Tensor<double, 3, 3>
//
//> MatrixTypes;
//
//TYPED_TEST_CASE(TestNMatrix, MatrixTypes);
////
////TYPED_TEST(TestNMatrix, swap){
////{
////
////	simpla::swap(TestFixture::vA,TestFixture::vB);
////
////	for (int i = 0; i < TestFixture::NDIMS; ++i)
////	for (int j = 0; j < TestFixture::MDIMS; ++j)
////	{
////		EXPECT_DOUBLE_EQ(0,abs(TestFixture::aA[i][j]-TestFixture::vB[i][j]));
////		EXPECT_DOUBLE_EQ(0,abs(TestFixture::aB[i][j]-TestFixture::vA[i][j]));
////	}
////
////}
////}
////
//TYPED_TEST(TestNMatrix, reduce){
//{
//	typename TestFixture::value_type expect=0;
//
//	for (int i = 0; i < TestFixture::NDIMS; ++i)
//	for (int j = 0; j < TestFixture::MDIMS; ++j)
//	{
//		expect+=TestFixture::aA[i][j]*TestFixture::aB[i][j];
//	}
//
//	auto value=simpla::inner_product(TestFixture::vA,TestFixture::vB);
//
//	EXPECT_DOUBLE_EQ(0,abs(expect -value));
//
//}
//}
//
//TYPED_TEST(TestNMatrix,Assign_Scalar){
//{
//
//	TestFixture::vD = TestFixture::a;
//
//	for (int i = 0; i < TestFixture::NDIMS; ++i)
//	for (int j = 0; j < TestFixture::MDIMS; ++j)
//
//	{
//		EXPECT_DOUBLE_EQ(0,abs(TestFixture::a-TestFixture::vD[i][j]) );
//	}
//}}
//
//TYPED_TEST(TestNMatrix,Assign_Array){
//{
//	TestFixture::vA = TestFixture::aA;
//
//	for (int i = 0; i < TestFixture::NDIMS; ++i)
//	for (int j = 0; j < TestFixture::MDIMS; ++j)
//	{
//		EXPECT_DOUBLE_EQ(0, abs(TestFixture::aA[i][j]-TestFixture::vA[i][j]));
//	}
//}}
//
//TYPED_TEST(TestNMatrix, Compare){
//{
//	TestFixture::vB = TestFixture::vA;
//
//	EXPECT_TRUE(TestFixture::vA == TestFixture::vB);
//	EXPECT_TRUE(TestFixture::vA != TestFixture::vC);
//	EXPECT_FALSE(TestFixture::vA == TestFixture::vC);
//	EXPECT_FALSE(TestFixture::vA != TestFixture::vB);
//
//}
//}
//
//TYPED_TEST(TestNMatrix, Arithmetic){
//{
//	TestFixture::vD =TestFixture::vA +TestFixture::a;
//
////	EQUATION(vA ,vB ,vC);
////
////	for (int i = 0; i < TestFixture::NDIMS; ++i)
////	for (int j = 0; j < TestFixture::MDIMS; ++j)
////	{
////		EXPECT_DOUBLE_EQ(0,abs(EQUATION(vA[i][j] ,vB[i][j] ,vC[i][j])- TestFixture::vD[i][j]));
////	}
//}
//}
//
//TYPED_TEST(TestNMatrix, performance_rawarray){
//{
//
//	for (std::size_t s = 0; s < 10000000L; ++s)
//	{
//		for(int i=0;i<TestFixture::NDIMS;++i)
//		for (int j = 0; j < TestFixture::MDIMS; ++j)
//		{	TestFixture::aD[i][j] +=EQUATION(aA[i][j] ,aB[i][j] ,aC[i][j])*s;};
//	}
//
//}
//}
//TYPED_TEST(TestNMatrix, performancenTuple){
//{
//
//	for (std::size_t s = 0; s < 10000000L; ++s)
//	{
//		TestFixture::vD +=EQUATION(vA ,vB ,vC)*(s);
//	}
//
//}
//}
