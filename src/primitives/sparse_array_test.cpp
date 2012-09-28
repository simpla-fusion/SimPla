/*
 * sparse_array_test.cpp
 *
 *  Created on: 2012-3-29
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <iostream>

#include "primitives/sparse_array.h"

using namespace simpla;

template<typename T>
class TestSparseArray: public testing::Test
{
protected:

	virtual void SetUp()
	{

	}

public:
	static const int NUM_OF_DIM_X = 10;
	static const int NUM_OF_DIM_Y = 20;
	static const int loop_num = 1000000L;
	typedef T ValueType;

	typedef SparseArray<ValueType> Vector;

};

typedef testing::Types<double /*, std::complex<double>*/> MyTypes;

TYPED_TEST_CASE(TestSparseArray, MyTypes);

TYPED_TEST(TestSparseArray, assign_vector){
{
	typename TestFixture::Vector v;

	for (size_t i = 0; i < TestFixture::NUM_OF_DIM_X; ++i)
	{
		v[i]=static_cast<typename TestFixture::ValueType>(i);
	}

	for (size_t i = 0; i < TestFixture::NUM_OF_DIM_X; ++i)
	{
		EXPECT_EQ(v[i], static_cast<typename TestFixture::ValueType>(i));
	}
}}

TYPED_TEST(TestSparseArray, arthimetic){
{
	typename TestFixture::Vector a,b,c;

	a[2] = 2;
	a[4] = 4;
	b[1] = 1;
	b[2] = 1.5;
	b[3] = 3;

	c= -a/4.0- b -2.0 + b/2.0+5.0;

	std::cout<<c<<std::endl;

	EXPECT_EQ(c.ZERO, static_cast<typename TestFixture::ValueType>(0));
	EXPECT_EQ(c[0], static_cast<typename TestFixture::ValueType>(c.ZERO));
	EXPECT_EQ(c[1], static_cast<typename TestFixture::ValueType>(-0.5));
	EXPECT_EQ(c[2], static_cast<typename TestFixture::ValueType>(-1.25));
	EXPECT_EQ(c[3], static_cast<typename TestFixture::ValueType>(-1.5));
	EXPECT_EQ(c[4], static_cast<typename TestFixture::ValueType>(-1 ));
	EXPECT_EQ(c[5], static_cast<typename TestFixture::ValueType>(c.ZERO));
	EXPECT_EQ(c[-1], static_cast<typename TestFixture::ValueType>(3 ));
}}
