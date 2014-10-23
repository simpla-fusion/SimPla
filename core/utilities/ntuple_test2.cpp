/*
 * ntuple_test2.cpp
 *
 *  Created on: 2014年10月23日
 *      Author: salmon
 */

#include <iostream>
#include "ntuple.h"
#include "sp_integer_sequence.h"
using namespace simpla;

int main(int argc, char **argv)
{

	nTuple<double, 2, 3> a = { 0, 1, 2, 3, 4, 5 };
	nTuple<double, 2, 3> b, c;

	std::cout << a << std::endl;

	b = 1;
	c = 2;
//
	a = b + c * 3;

//	std::cout << seq_reduce(integer_sequence<size_t, 2, 3>(), _impl::plus(), a)
//			<< std::endl;

//	std::cout << _impl::get_value_traits(a.data_, 1, 2) << std::endl;

	std::cout << a << std::endl;

	std::cout << b << std::endl;

	std::cout << c << std::endl;

////	nTuple<double, 4, 5> b;
////	nTuple<double, 4, 5> c;
////	a = 1.0;
////	b = 2.0;
////	c = a * 3.0 + b * 2;
////	c = 1;
//
//	size_t id[2] = { 1, 2 };
//
//	double d[2][3] = { 0, 1, 2, 3, 4, 5 };
//
//	std::cout << std::boolalpha << std::is_array<decltype(d )>::value
//			<< std::endl;
////	std::cout << typeid(decltype(d)).name() << std::endl;
////
//	std::cout << get_value2(d, id) << std::endl;
////	std::cout << get_value2(a, id) << std::endl;
//
////	std::cout << typeid(decltype( get_value2(a.data_, id) )).name()
////			<< std::endl;
//
////	std::cout << typeid(decltype(c.data_)).name() << std::endl;
//
////	seq_for_each(i_seq(), [&](size_t const idx[2])
////	{
////		std::cout<<get_value(d,idx)<<std::endl;
//////			std::cout<<get_value(c,idx)<<std::endl;
////
////		});

//	std::cout << _seq_reduce<3>::eval(_impl::plus(), c) << std::endl;
//
//	std::cout << std::boolalpha
//			<< bool(_seq_reduce<3>::eval(_impl::logical_and(), a == b))
//			<< std::endl;
}
