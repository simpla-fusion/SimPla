/*
 * ntuple_test2.cpp
 *
 *  Created on: 2014-10-23
 *      Author: salmon
 */

#include <iostream>

#include "../integer_sequence.h"
#include "../nTuple.h"
#include "../nTupleExt.h"

using namespace simpla;

int main(int argc, char **argv)
{

    nTuple<double, 3> b = {1, 2, 3,}, c = {0, 1, 2,};
//
//	std::cout << a << std::endl;
//
//	b = 1;
//	c = 2;
//
//	std::cout << b << std::endl;
//
//	std::cout << c << std::endl;

    auto t = 0.3 * (b - c) + c;
//
//	std::cout << std::boolalpha
//			<< std::is_same<decltype(t)::value_type, double>::value << std::endl
//			<< std::is_same<decltype(t)::dimensions, integer_sequence<size_t, 3>>::value
//			<< std::endl
//			<< std::is_same<decltype(t)::primary_type, nTuple<double, 3>>::value
//			<< std::endl << std::endl;

//
    nTuple<double, 3> a = 0.3 * (b - c) + c;

//	std::cout << seq_reduce(integer_sequence<size_t, 2, 3>(), _impl::plus(), a)
//			<< std::endl;

//	std::cout << _impl::traits::index_traits(a.data_, 1, 2) << std::endl;
//
    std::cout << a << std::endl;

    nTuple<double, 5> d;
    d = 10;
    d = a;
    std::cout << d << std::endl;
//
//	std::cout << b << std::endl;
//
//	std::cout << c << std::endl;
//
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
//	std::cout << traits::index2(d, id) << std::endl;
////	std::cout << traits::index2(a, id) << std::endl;
//
////	std::cout << typeid(decltype( traits::index2(a.data_, id) )).name()
////			<< std::endl;
//
////	std::cout << typeid(decltype(c.data_)).name() << std::endl;
//
////	seq_for_each(i_seq(), [&](size_t const idx[2])
////	{
////		std::cout<<traits::index(d,idx)<<std::endl;
//////			std::cout<<traits::index(c,idx)<<std::endl;
////
////		});

//	std::cout << _seq_reduce<3>::eval(_impl::plus(), c) << std::endl;
//
//	std::cout << std::boolalpha
//			<< bool(_seq_reduce<3>::eval(_impl::logical_and(), a == b))
//			<< std::endl;
}
