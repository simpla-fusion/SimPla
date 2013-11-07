/*
 * test.cpp
 *
 *  Created on: 2013年11月1日
 *      Author: salmon
 */

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

struct CCC
{
	static void Foo(double a, int, int)
	{
		std::cout << "First" << a << std::endl;
	}

	static void Foo(double b, int)
	{
		std::cout << "Second" << b << std::endl;
	}

};
template<typename Fun, typename ... Args>
void foo(Fun const &fun, Args ... args)
{
	fun(args...);
}

template<typename Fun, typename ...Args>
void Foo(Fun fun, Args & ... args)
{
	[&](Args & ... args)
	{
		fun(args...);
	}
}

template<typename ... Args>
void foo2(Args ... args)
{
	foo<void(double, Args...)>(CCC::Foo, 3.0, args...);
}
int main()
{
	for (int i = 0; i < 10; ++i)
	{

		[&](Args ... args)
		{

		}

	}

}

