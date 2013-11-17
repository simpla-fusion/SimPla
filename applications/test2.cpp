/*
 * test2.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#include <iostream>

template<int ...N>
struct Foo
{
	void foo()
	{
		std::cout << "This is NULL!" << std::endl;
	}
};

template<int M, int ...N>
struct Foo<M, N...> : public Foo<N...>
{
	void foo()
	{
		std::cout << "This is " << M << std::endl;
		Foo<N...>::foo();
	}
};

template<int ...N>
struct Foo<0, N...> : public Foo<N...>
{
	void foo()
	{
		std::cout << "I'm Zero!" << std::endl;
		Foo<N...>::foo();
	}
};

int main(int argc, char **argv)
{
	Foo<1, 0, 3, 4> tmp;
	tmp.foo();
}
