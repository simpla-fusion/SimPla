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

void foo(int a, const std::string& b, float c)
{
	std::cout << a << " , " << b << " , " << c << '\n';
}

int main()
{
	std::cout << std::boolalpha
			<<

			std::is_same<typename traits::extents<nTuple<int, 3, 4> >::type,
					typename traits::extents<int[3][4]>::type>::value

			<< std::endl;
}
