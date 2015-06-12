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

	typedef typename traits::extents<type>::type extents;

	nTuple<std::size_t, traits::extent<extents>::value> DIMENSIONS;

	typedef typename type::value_type value_type;

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

TYPED_TEST(TestNtuple, reduce){
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

TYPED_TEST(TestNtuple, equation ){
{
	TestFixture::vB=TestFixture::vA+1000;

	EXPECT_TRUE( TestFixture::vA==TestFixture::vA);
	EXPECT_TRUE( TestFixture::vA!=TestFixture::vB);
//	EXPECT_TRUE( TestFixture::vA<=TestFixture::vA);
//	EXPECT_TRUE( TestFixture::vA<=TestFixture::vB);
//	EXPECT_TRUE( TestFixture::vA<TestFixture::vB);
//	EXPECT_TRUE( TestFixture::vA>=TestFixture::vA);
//	EXPECT_TRUE( TestFixture::vB>TestFixture::vA);

	EXPECT_FALSE( TestFixture::vA!=TestFixture::vA);
	EXPECT_FALSE( TestFixture::vA==TestFixture::vB);
//	EXPECT_FALSE( TestFixture::vA>=TestFixture::vB);
//	EXPECT_FALSE( TestFixture::vB<=TestFixture::vA);
//	EXPECT_FALSE( TestFixture::vA>TestFixture::vB);
// 	EXPECT_FALSE( TestFixture::vB<TestFixture::vA);

}
}
//

TYPED_TEST(TestNtuple, inner_product){
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

template<typename > class nTuplePerf1;

template<typename T, size_t N>
class nTuplePerf1<nTuple<T, N>> : public testing::Test
{
	virtual void SetUp()
	{

		a = 1;
		b = 3;
		c = 4;
		d = 7;

		dims0 = N;

		for (int i = 0; i < dims0; ++i)
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

	}
public:
	std::size_t num_of_loops = 10000000L;

	typedef nTuple<T, N> type;

	typedef T value_type;

	type vA, vB, vC, vD;

	value_type aA[N], aB[N], aC[N], aD[N], res[N];

	value_type a, b, c, d;

	size_t dims0, dims1;

};
typedef testing::Types<

nTuple<double, 3>

, nTuple<double, 20>

, nTuple<int, 3>

, nTuple<int, 10>

, nTuple<std::complex<double>, 3>

, nTuple<std::complex<double>, 10>

> ntuple_type_lists_1d;

TYPED_TEST_CASE(nTuplePerf1, ntuple_type_lists_1d);

TYPED_TEST(nTuplePerf1,performance_raw_array){
{
	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
	{
		for (int i = 0; i < TestFixture::dims0; ++i)
		{
			TestFixture::aD[i] += EQUATION( TestFixture::aA[i], TestFixture::aB[i], TestFixture::aC[i]) * s;
		}

	}
}
}

TYPED_TEST(nTuplePerf1, performancenTuple){
{

	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
	{
		TestFixture::vD +=EQUATION(TestFixture::vA ,TestFixture::vB ,TestFixture::vC)*(s);
	}

}
}

template<typename > class nTuplePerf2;

template<typename T, size_t N, size_t M>
class nTuplePerf2<nTuple<T, N, M>> : public testing::Test
{
	virtual void SetUp()
	{

		a = 1;
		b = 3;
		c = 4;
		d = 7;

		dims0 = N;
		dims1 = M;

		for (int i = 0; i < dims0; ++i)
			for (int j = 0; j < dims1; ++j)
			{
				aA[i][j] = i * 2;
				aB[i][j] = 5 - i;
				aC[i][j] = i * 5 + 1;
				aD[i][j] = 0;
				vA[i][j] = aA[i][j];
				vB[i][j] = aB[i][j];
				vC[i][j] = aC[i][j];
				vD[i][j] = 0;

				res[i][j] = -(aA[i][j] + a) / (aB[i][j] * b - c) - aC[i][j];
			}

	}
public:
	std::size_t num_of_loops = 10000000L;

	typedef nTuple<T, M, N> type;

	typedef T value_type;

	type vA, vB, vC, vD;

	value_type aA[N][M], aB[N][M], aC[N][M], aD[N][M], res[N][M];

	value_type a, b, c, d;

	size_t dims0, dims1;

};
typedef testing::Types<

nTuple<double, 3, 4>

, nTuple<double, 20, 10>

, nTuple<int, 3, 4>

, nTuple<int, 10, 20>

, nTuple<std::complex<double>, 3, 4>

, nTuple<std::complex<double>, 10, 20>

> ntuple_type_lists_2d;

TYPED_TEST_CASE(nTuplePerf2, ntuple_type_lists_2d);

TYPED_TEST(nTuplePerf2,performance_raw_array){
{
	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
	{
		for (int i = 0; i < TestFixture::dims0; ++i)
		for (int j = 0; j < TestFixture::dims1; ++j)
		{
			TestFixture::aD[i][j] += EQUATION( TestFixture::aA[i][j], TestFixture::aB[i][j], TestFixture::aC[i][j]) * s;
		}

	}
}
}

TYPED_TEST(nTuplePerf2, performancenTuple){
{

	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
	{
		TestFixture::vD +=EQUATION(TestFixture::vA ,TestFixture::vB ,TestFixture::vC)*(s);
	}

}
}

