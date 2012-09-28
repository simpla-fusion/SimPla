/*
 * sparse_matrix_test.cpp
 *
 *  Created on: 2012-3-28
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <iostream>

#include "primitives/sparse_matrix.h"

using namespace simpla;

template<typename T>
class TestSparseMatrix: public testing::Test
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

	typedef SparseVector<ValueType> Vector;
	typedef SparseMatrix<ValueType> Matrix;

};

typedef testing::Types<double, std::complex<double> > MyTypes;

TYPED_TEST_CASE(TestSparseMatrix, MyTypes);

TYPED_TEST(TestSparseMatrix, assign_vector){
{
	typename TestFixture::Vector v;

	for (size_t i = 0; i < TestFixture::NUM_OF_DIM_X; ++i)
	{
		v[i]=static_cast<typename TestFixture::ValueType>(i);
	}

	std::cout<< v <<std::endl;

	for (size_t i = 0; i < TestFixture::NUM_OF_DIM_X; ++i)
	{
		EXPECT_EQ(v[i], static_cast<typename TestFixture::ValueType>(i));
	}
}}

TYPED_TEST(TestSparseMatrix, assign_matrix){
{
	typename TestFixture::Matrix m;

	for (size_t i = 0; i < TestFixture::NUM_OF_DIM_X; i+=2)
	for (size_t j = 0; j < TestFixture::NUM_OF_DIM_Y; j+=4)
	{
		m[i][j]=static_cast<typename TestFixture::ValueType>(i*TestFixture::NUM_OF_DIM_Y+j);
	}
	std::cout<< m <<std::endl;
	for (size_t i = 0; i < TestFixture::NUM_OF_DIM_X; i+=2)
	for (size_t j = 0; j < TestFixture::NUM_OF_DIM_Y; j+=4)
	{
		EXPECT_EQ(m[i][j],static_cast<typename TestFixture::ValueType>(i*TestFixture::NUM_OF_DIM_Y+j));
	}

}}
