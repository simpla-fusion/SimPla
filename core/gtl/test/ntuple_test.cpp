/**
 * @file ntuple_test.cpp
 *
 *  created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include "../type_traits.h"
#include "../ntuple.h"
#include "../primitives.h"

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

		DIMENSIONS = extents::value;

		_impl::seq_for_each(extents(),

		[&](size_t const idx[traits::extent<extents>::value])
		{
			try_index_r(aA,idx) = idx[0] * 2;
			try_index_r(aB,idx) = 5 - idx[0];
			try_index_r(aC,idx) = idx[0] * 5 + 1;
			try_index_r(aD,idx) = 0;
			try_index_r(vA,idx) = try_index_r(aA,idx);
			try_index_r(vB,idx) = try_index_r(aB,idx);
			try_index_r(vC,idx) = try_index_r(aC,idx);
			try_index_r(vD,idx) = 0;

			try_index_r(res,idx) = -(try_index_r(aA,idx) + a) /
			(try_index_r(aB,idx) * b - c) - try_index_r(aC,idx);

		});

		num_of_loops = 1000000L;
	}

public:

	std::size_t num_of_loops = 10000000L;

	typedef T type;

	typedef traits::extents_t<type> extents;

	nTuple<std::size_t, traits::extent<extents>::value> DIMENSIONS;

	typedef traits::value_type_t<type> value_type;

	type vA, vB, vC, vD;

	typename traits::pod_type<T>::type aA, aB, aC, aD, res;

	value_type a, b, c, d;

};

typedef testing::Types<

nTuple<double, 3>

, nTuple<double, 3, 3>

, nTuple<double, 3, 4, 5>

, nTuple<int, 3, 4, 5, 6>

, nTuple<std::complex<double>, 3, 4, 5, 6>

> ntuple_type_lists;

TYPED_TEST_CASE(TestNtuple, ntuple_type_lists);

TYPED_TEST(TestNtuple, swap){
{

	swap(TestFixture::vA, TestFixture::vB);

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				EXPECT_DOUBLE_EQ(0, std::abs(try_index_r(TestFixture::aA,idx)- try_index_r(TestFixture:: vB,idx)));
				EXPECT_DOUBLE_EQ(0, std::abs(try_index_r(TestFixture::aB,idx)- try_index_r(TestFixture:: vA,idx)));
			});

}
}
TYPED_TEST(TestNtuple, assign_Scalar){
{

	TestFixture::vA = TestFixture::a;

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				EXPECT_DOUBLE_EQ(0, abs(TestFixture::a- try_index_r(TestFixture:: vA,idx)));
			}
	);

}}

TYPED_TEST(TestNtuple, assign_Array){
{
	TestFixture::vA = TestFixture::aA;

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				EXPECT_DOUBLE_EQ(0, abs(try_index_r(TestFixture::aA,idx)- try_index_r(TestFixture:: vA,idx)));
			}
	);

}}
TYPED_TEST(TestNtuple, self_assign){
{
	TestFixture::vB +=TestFixture::vA;

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				EXPECT_DOUBLE_EQ(0,abs( try_index_r(TestFixture::vB,idx)
								- (try_index_r(TestFixture::aB,idx)+
										try_index_r(TestFixture::aA,idx))));

			}
	);

}
}

TYPED_TEST(TestNtuple, arithmetic){
{
	TestFixture::vD = EQUATION(TestFixture::vA, TestFixture::vB, TestFixture::vC);

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				auto &ta=try_index_r(TestFixture::vA,idx);
				auto &tb=try_index_r(TestFixture::vB,idx);
				auto &tc=try_index_r(TestFixture::vC,idx);
				auto &td=try_index_r(TestFixture::vD,idx);

				EXPECT_DOUBLE_EQ(0, abs(EQUATION(ta,tb,tc ) - td));
			}
	);

}
}

TYPED_TEST(TestNtuple, cross){
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
	vD-=vC;
	EXPECT_DOUBLE_EQ(0,abs(vD[0])+abs(vD[1])+abs(vD[2]) );
}}

