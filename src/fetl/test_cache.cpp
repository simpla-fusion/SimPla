/*
 * test_cache.cpp
 *
 *  Created on: 2014年5月1日
 *      Author: salmon
 */

#include "cache.h"
#include "../utilities/log.h"
#include <iostream>
#include <new>
#include <string>
#include <type_traits>
#include <typeinfo>

using namespace simpla;

struct Foo
{
	Foo()
	{
		CHECK("Constructor") << std::endl;
	}
	~Foo()
	{
		CHECK("Destructor") << std::endl;
	}
};

int main(int argc, char **argv)
{

	Foo foo;
	Foo * pfoo = new Foo();
	auto a = Cache<Foo>(foo);
	auto b = Cache<Foo*>(pfoo);

	std::cout << std::boolalpha << std::is_reference<decltype(*a)>::value
			<< std::endl

			<< std::boolalpha << std::is_reference<decltype(*b)>::value
			<< std::endl;
	std::cout << typeid(*a).name() << " " << typeid(*b).name() << std::endl;

	delete pfoo;
}

