/*
 * test1.cpp
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>

class Foo
{
public:
	Foo()
	{
		std::cout << "Default Construct" << std::endl;
	}
	Foo(Foo const &)
	{
		std::cout << "Copy Construct" << std::endl;
	}
	~Foo()
	{
		std::cout << "Destroy" << std::endl;
	}
	int a;
private:
};

template<typename Args>
void foo(Args &&args)
{
	std::cout << typeid(args).name() << "  " << args.a << std::endl;
	++args.a;

}
int main(int argc, char **argv)
{
	Foo a;
	a.a = 1;
	std::cout << "======================" << std::endl;
	foo(std::forward<Foo>(a));
	std::cout << "======================" << std::endl;
	foo(std::forward<Foo &>(a));
	std::cout << "======================" << std::endl;
//	foo(&a);
//	std::cout << "======================" << std::endl;
}
