/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include <iostream>
#include "../../core/gtl/type_traits.h"
#include "../../core/gtl/ntuple.h"

using namespace simpla;

int main()
{
	nTuple<double, 3> foo = { 1, 2, 3 };
	std::cout << traits::get<0>(foo) << " " << traits::get<1>(foo) << " "
			<< traits::get<2>(foo) << " " << std::endl;
}
