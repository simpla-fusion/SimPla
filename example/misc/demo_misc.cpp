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
	void foo(int a)
	{
		std::cout << "This is U! a= [" << a << "]" << std::endl;
	}

};

struct W
{
	void foo(int a)
	{
		std::cout << "This is W! a= [" << a << "]" << std::endl;
	}

};

int main()
{

	Signal<void(int)> sig;

	auto u = std::make_shared<U>();
	auto w = std::make_shared<W>();

	sig.connect(u, &U::foo);

	auto it_u = sig.connect(w, &W::foo);

	sig.connect([](int i)
	{
	    std::cout << "This is lambda! a= [" << i << "]" << std::endl;

	});

	sig(10);

	sig.disconnect(it_u);

	std::cout<<"======================"<<std::endl;

	sig(12);

}
