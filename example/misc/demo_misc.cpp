/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include <iostream>
#include "../../core/gtl/type_traits.h"

using namespace simpla;

void foo(int a, const std::string& b, float c)
{
	std::cout << a << " , " << b << " , " << c << '\n';
}

int main()
{
	auto args = std::make_tuple(2, "Hello", 3.5);
	invoke(foo, args);
}
