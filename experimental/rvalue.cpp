/*
 * rvalue.cpp
 *
 *  Created on: 2014年10月22日
 *      Author: salmon
 */

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <typeinfo>

struct Foo
{
	Foo()
	{
		std::cout << "construct" << std::endl;
	}

	Foo(Foo const &)
	{
		std::cout << "copy construct" << std::endl;
	}

	Foo(Foo &&)
	{
		std::cout << "move construct" << std::endl;
	}
	~Foo()
	{
		std::cout << "destroy" << std::endl;
	}
};

template<typename T>
struct P2
{
	typedef typename std::remove_reference<T>::type type;
	static constexpr bool pass_value = std::is_rvalue_reference<T>::value;

	typedef typename std::conditional<pass_value, type &&, type const &>::type param_type;
	typedef typename std::conditional<pass_value, type, type const &>::type storage_type;

	storage_type foo_;

	P2(param_type f) :
			foo_(f)
	{
		std::cout << std::boolalpha << pass_value << std::endl;
		std::cout << typeid(T).name() << std::endl;
		std::cout << typeid(storage_type).name() << std::endl;
	}
	~P2()
	{
	}
};

template<typename T>
P2<T> foo1(T && f)
{
	return P2<T>(std::forward<T>(f));
}
//
//void foo2(Foo && f)
//{
//
//}
//
//template<typename Args>
//void foo3(Args && args)
//{
//	foo1(std::forward<Args>(args));
//}
//
//template<typename Args>
//void foo4(Args && args)
//{
//	foo2(std::forward<Args>(args));
//}
int main(int argc, char **argv)
{
	double f;

//	auto p1 = foo1(f);
	auto p2 = foo1(std::move(f));
}
