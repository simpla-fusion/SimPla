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
#include "../../core/gtl/design_pattern/signal.h"

using namespace simpla;

struct U
{
	int foo(int a)
	{
		std::cout << "This is U! a= [" << a << "]" << std::endl;
		return 1;
	}

};

struct W
{
	int foo(int a)
	{
		std::cout << "This is W! a= [" << a << "]" << std::endl;
		return 2;
	}

};

int main()
{

	Signal<double(int)> sig;

	auto u = std::make_shared<U>();
	auto w = std::make_shared<W>();

	sig.connect(u, &U::foo);

	auto it_u = sig.connect(w, &W::foo);

	sig.connect([](int i)
	{
	    std::cout << "This is lambda! a= [" << i << "]" << std::endl;
	    return 3;
	});

	sig(10);

	{
		auto res = sig(12);

		std::cout << res[0] << " " << res[1] << " " << res[2] << " " << std::endl;
	}
	sig.disconnect(it_u);

	std::cout << "======================" << std::endl;
	{
		auto res = sig(12);

		std::cout << res[0] << " " << res[1] << " " << res[2] << " " << std::endl;
	}

}
