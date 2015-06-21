/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include <iostream>
#include <typeinfo>
#include <tuple>
#include <complex>
#include "../../core/gtl/mpl.h"
#include "../../core/gtl/ntuple.h"
using namespace simpla;
int main()
{
	nTuple<std::complex<double>, 3, 4, 5, 6, 7> a, b, c;

//	std::cout << typeid(decltype( a.at(0 ) )).name() << std::endl;
//	std::cout << typeid(decltype( a.at(0, 1 ) )).name() << std::endl;
//	std::cout << typeid(decltype( a.at(0, 1, 2) )).name() << std::endl;
//	std::cout << typeid(decltype( a.at(0, 1, 2, 3) )).name() << std::endl;

	a = b + b;
}
