/*
 * testnTuple.cpp
 *
 *  created on: 2012-1-10
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "ntuple.h"

#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include "expression_template.h"
//#include "log.h"
//#include "pretty_stream.h"

using namespace simpla;

int main(int argc, char **argv)
{

	double a[4] = { 0, 0, 0, 0 };
	nTuple<double, 4> b = { 0, 0, 0, 0 };
//	get_value(a, integer_sequence<std::size_t, 2>()) = 10;
//	get_value(b, integer_sequence<std::size_t, 2ul>()) = 12;
//	b[3] = 24;
//
//	std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << std::endl;
//	std::cout << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << std::endl;
//	std::cout
//			<< typeid(decltype(get_value(b, integer_sequence<std::size_t , 2>()))).name()
//			<< std::endl;
//
//	get_value(b.data_, integer_sequence<std::size_t, 2>()) = 12;
//	std::cout << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << std::endl;
//
	nTuple<double, 3, 4> m = { 2, 3, 4, 5, 5, 6, 6, 7, 7, 8 };
	nTuple<double, 3, 4> n = { 2, 3, 4, 5, 5, 6, 6, 7, 7, 8 };

	m = 2;

	n = (m + 1);

	std::cout << n[2][3] << std::endl;

//	auto value = inner_product(m, n);
//	std::cout << value << std::endl;

//
//	nTuple<double, 3, 4> n;
//
//	std::cout << m[index_sequence<0, 0>()] << std::endl;
//	m[index_sequence<0, 1>()] = 3.15;
//	std::cout << m[index_sequence<0, 1>()] << std::endl;
//	m[1][2] = 10;
//	std::cout << m[index_sequence<1, 2>()] << std::endl;
//
//	std::cout << m << std::endl;
//
//	n = 2;
//
//	m -= n;
	std::cout << m << std::endl;
	std::cout << n << std::endl;
//
//	n = 2;
//
//	m = 2;
//
//	auto v = inner_product(m, n);
//
//	std::cout << v << std::endl;
//	std::cout << std::boolalpha << bool((m == n)) << std::endl;
//	m = 3;
	std::cout << std::boolalpha << bool((m == n)) << std::endl;
	m += 1;
	std::cout << std::boolalpha << bool((m == n)) << std::endl;

	std::cout << m << std::endl;
	std::cout << n << std::endl;
//
//	swap(m, n);
//
//	std::cout << m << std::endl;
//	std::cout << n << std::endl;

}

//#define EQUATION(_A,_B,_C)  ( -(TestFixture::_A  +TestFixture::a )/(   TestFixture::_B *TestFixture::b -TestFixture::c  )- TestFixture::_C)
//
//template<typename T>
//class TestNtuple: public testing::Test
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
//
//		DIMENSIONS = seq2ntuple(typename nTuple_traits<T>::dimensions());
//
//		for (int i = 0; i < DIMENSIONS[0]; ++i)
//		{
//			aA[i] = i * 2;
//			aB[i] = 5 - i;
//			aC[i] = i * 5 + 1;
//			aD[i] = 0;
//			vA[i] = aA[i];
//			vB[i] = aB[i];
//			vC[i] = aC[i];
//			vD[i] = 0;
//
//			res[i] = -(aA[i] + a) / (aB[i] * b - c) - aC[i];
//
//		}
//
//		num_of_loops = 1000000L;
//	}
//
//public:
//
//	typedef T type;
//
//	nTuple<std::size_t, nTuple_traits<T>::dimensions::size()> DIMENSIONS;
//
//	typedef typename T::value_type value_type;
//
//	std::size_t num_of_loops = 10000000L;
//
//	T vA, vB, vC, vD;
//	typename T::data_type aA, aB, aC, aD, res;
//	value_type a, b, c, d;
//
//};
//
//typedef testing::Types<
//
//nTuple<double, 3>
//
////, nTuple<double, 10>
////
////, nTuple<double, 20>
////
////, nTuple<int, 3>
////
////, nTuple<int, 10>
////
////, nTuple<int, 20>
////
////, nTuple<std::complex<double>, 3>
////
////, nTuple<std::complex<double>, 10>
////
////, nTuple<std::complex<double>, 20>
//
//> nTupleTypes;
//
//TYPED_TEST_CASE(TestNtuple, nTupleTypes);
//
//TYPED_TEST(TestNtuple, swap){
//{
//
//	simpla::swap(TestFixture::vA, TestFixture::vB);
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		EXPECT_DOUBLE_EQ(0, abs(TestFixture::aA[i] - TestFixture::vB[i]));
//		EXPECT_DOUBLE_EQ(0, abs(TestFixture::aB[i] - TestFixture::vA[i]));
//	}
//}
//}
//
//TYPED_TEST(TestNtuple, reduce){
//{
//	typename TestFixture::value_type expect=0;
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		expect+=TestFixture::aA[i];
//	}
//
//	std::cout<<TestFixture::vA<<std::endl;
//	std::cout<<TestFixture::vB<<std::endl;
//
//	auto value=seq_reduce(typename nTuple_traits<typename TestFixture::type>::dimensions() , _impl::plus(), TestFixture::vA);
////			seq_reduce(typename nTuple_traits<typename TestFixture::type>::dimensions() ,_impl::plus(),TestFixture::vA);
//
//	std::cout<< value<<std::endl;
//
////	inner_product(TestFixture::vA,TestFixture::vB);
//
//	std::cout<<expect<<std::endl;
//	std::cout<<value<<std::endl;
//	EXPECT_DOUBLE_EQ(0,abs(expect -value));
//
//}
//}
//
//TYPED_TEST(TestNtuple, Assign_Scalar){
//{
//
//	TestFixture::vD = TestFixture::aA;
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		EXPECT_DOUBLE_EQ(0,abs(TestFixture::aA[i]-TestFixture::vD[i]) );
//	}
//}}
//
//TYPED_TEST(TestNtuple, Assign_Array){
//{
//	TestFixture::vA = TestFixture::aA;
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		EXPECT_DOUBLE_EQ( abs(TestFixture::aA[i]), abs(TestFixture::vA[i]));
//	}
//}}
//
//TYPED_TEST(TestNtuple, Arithmetic){
//{
//	TestFixture::vD = EQUATION(vA ,vB ,vC);
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		EXPECT_DOUBLE_EQ(0,abs(EQUATION(vA[i] ,vB[i] ,vC[i])- TestFixture::vD[i]));
//	}
//}
//}
//
//TYPED_TEST(TestNtuple, self_assign){
//{
//	TestFixture::vB +=TestFixture::vA;
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		EXPECT_DOUBLE_EQ(0,abs( TestFixture::vB[i] -(TestFixture::aB[i]+ TestFixture::aA[i])));
//	}
//}
//}
//
//TYPED_TEST(TestNtuple, compare){
//{
//	EXPECT_TRUE( TestFixture::vA==TestFixture::aA);
//	EXPECT_FALSE( TestFixture::vA==TestFixture::vB);
//
//	EXPECT_TRUE( TestFixture::vA!=TestFixture::vB);
//	EXPECT_FALSE( TestFixture::vA!=TestFixture::vA);
//
//}
//}
//
//TYPED_TEST(TestNtuple, Dot){
//{
//	typename TestFixture::value_type res;
//
//	res=0;
//
//	for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
//	{
//		res += TestFixture::vA[i] * TestFixture::vB[i];
//	}
//	EXPECT_DOUBLE_EQ(abs(res),abs( dot( TestFixture::vA, TestFixture::vB)));
//}}
//
//TYPED_TEST(TestNtuple, Cross){
//{
//	nTuple< typename TestFixture::value_type,3> vA, vB,vC ,vD;
//
//	for (int i = 0; i < 3; ++i)
//	{
//		vA[i] = (i * 2);
//		vB[i] = (5 - i);
//	}
//
//	for (int i = 0; i < 3; ++i)
//	{
//		vD[i] = vA[(i + 1) % 3] * vB[(i + 2) % 3]
//		- vA[(i + 2) % 3] * vB[(i + 1) % 3];
//	}
//
//	vC=cross(vA,vB);
//
//	EXPECT_EQ(vD ,vC);
//}}
//
//TYPED_TEST(TestNtuple, performance_rawarray){
//{
//	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
//	{
//		for(int i=0;i<TestFixture::DIMENSIONS[0];++i)
//		{	TestFixture::aD[i] +=EQUATION(aA[i] ,aB[i] ,aC[i])*s;};
//	}
//
////		for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
////	{
////		EXPECT_DOUBLE_EQ(abs(EQUATION(aA[i] ,aB[i] ,aC[i])),abs(TestFixture::aD[i]/TestFixture::m) );
////	}
//
//}
//}
//TYPED_TEST(TestNtuple, performancenTuple){
//{
//
//	for (std::size_t s = 0; s < TestFixture::num_of_loops; ++s)
//	{
//		TestFixture::vD +=EQUATION(vA ,vB ,vC)*(s);
//	}
//
////		for (int i = 0; i < TestFixture::DIMENSIONS[0]; ++i)
////	{
////		EXPECT_DOUBLE_EQ(abs(EQUATION(vA[i] ,vB[i] ,vC[i])) ,abs(TestFixture::vD[i]/TestFixture::m));
////	}
//}
//}

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
//	static constexpr std::size_t   NDIMS = seq_get_value<0,
//			typename nTuple_traits<T>::dimensions>::value;
//
//	static constexpr std::size_t   MDIMS = seq_get_value<1,
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
//Matrix<double, 3u, 3u>
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
//	for (std::size_t   s = 0; s < 10000000L; ++s)
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
//	for (std::size_t   s = 0; s < 10000000L; ++s)
//	{
//		TestFixture::vD +=EQUATION(vA ,vB ,vC)*(s);
//	}
//
//}
//}
