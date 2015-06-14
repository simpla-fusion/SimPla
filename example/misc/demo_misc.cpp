/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include <iostream>
#include "../../core/gtl/type_traits.h"
#include "../../core/gtl/ntuple.h"
#include "../../core/gtl/ntuple_ext.h"

using namespace simpla;

int main()
{
	nTuple<double, 3, 2> foo = { 1, 2, 3, 4, 5, 6 };
	std::cout << traits::get<0, 1>(foo) << " " << traits::get<1, 2>(foo) << " "
			<< traits::get<1, 1>(foo) << " " << std::endl;

	std::cout << foo << std::endl;
}
