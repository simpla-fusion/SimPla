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
	nTuple<std::complex<double>, 3> a, b;

	a = b + 2;
}
