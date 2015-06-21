/**
 * @file ntuple_reduce_test.cpp
 *
 *  Created on: 2015年6月21日
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

template<typename T>
class TestNtupleReduce: public testing::Test
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

//, nTuple<double, 3, 3>
//
//, nTuple<double, 3, 4, 5>
//
//, nTuple<int, 3, 4, 5, 6>
//
//, nTuple<std::complex<double>, 3, 4, 5, 6>

> ntuple_type_lists;

TYPED_TEST_CASE(TestNtupleReduce, ntuple_type_lists);

//TYPED_TEST(TestNtupleReduce, equation ){
//{
//	TestFixture::vB=TestFixture::vA+1000;
//
//	EXPECT_TRUE( TestFixture::vA==TestFixture::vA);
//	EXPECT_TRUE( TestFixture::vA!=TestFixture::vB);
////	EXPECT_TRUE( TestFixture::vA<=TestFixture::vA);
////	EXPECT_TRUE( TestFixture::vA<=TestFixture::vB);
////	EXPECT_TRUE( TestFixture::vA<TestFixture::vB);
////	EXPECT_TRUE( TestFixture::vA>=TestFixture::vA);
////	EXPECT_TRUE( TestFixture::vB>TestFixture::vA);
//
//	EXPECT_FALSE( TestFixture::vA!=TestFixture::vA);
//	EXPECT_FALSE( TestFixture::vA==TestFixture::vB);
////	EXPECT_FALSE( TestFixture::vA>=TestFixture::vB);
////	EXPECT_FALSE( TestFixture::vB<=TestFixture::vA);
////	EXPECT_FALSE( TestFixture::vA>TestFixture::vB);
//// 	EXPECT_FALSE( TestFixture::vB<TestFixture::vA);
//
//}
//}
TYPED_TEST(TestNtupleReduce , reduce){
{
	typename TestFixture::value_type expect=0;

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				expect+=try_index_r(TestFixture::vA,idx);
			}
	);
	auto value=seq_reduce(typename TestFixture::extents(),_impl::plus(), TestFixture::vA);

	EXPECT_DOUBLE_EQ(0,abs(expect -value));

}
}
TYPED_TEST(TestNtupleReduce, inner_product){
{
	typename TestFixture::value_type res;

	res=0;

	_impl::seq_for_each(typename TestFixture::extents(),
			[&](size_t const idx[traits::extent<typename TestFixture::extents,0>::value])
			{
				res += try_index_r(TestFixture::aA,idx) * try_index_r(TestFixture::aB,idx);
			}
	);

	EXPECT_DOUBLE_EQ(0,abs(res- inner_product( TestFixture::vA, TestFixture::vB)));
}}
