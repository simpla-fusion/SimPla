/*
 * test_visitor.cpp
 *
 *  Created on: 2013年12月22日
 *      Author: salmon
 */
#include <tuple>
#include <iostream>
#include <memory>
#include "../src/utilities/type_utilites.h"

using namespace simpla;

struct Foo1: public AcceptorBase
{
	typedef Foo1 this_type;

	virtual bool CheckType(std::type_info const &t_info)
	{
		return typeid(this_type) == t_info;
	}

	template<typename ...Args>
	void accept(Visitor<this_type, Args...> &visitor)
	{
		visitor.excute([this](Args ... args)
		{	this->Command(std::forward<Args>(args)...);});
	}
	void accept(Visitor<this_type, const char *> &visitor)
	{
		if (visitor.GetName() == "Command2")
		{
			visitor.excute([this](std::string const & args)
			{	this->Command2(args);});
		}
		else
		{
			std::cout << "Unknown function name!" << std::endl;
		}
	}

	void Command2(std::string const & s)
	{
		std::cout << "This is Foo1::Command2(string). args=" << s << std::endl;
	}

	void Command(int a, int b)
	{
		std::cout << "This is Foo1::Command(int,int). args=" << a << "     " << b << std::endl;
	}

	template<typename ... Args>
	void Command(Args const & ...args)
	{
		std::cout << "This is Foo1::Command(args...). args=";

		Print(args...);

		std::cout << std::endl;
	}

	void Print()
	{
	}

	template<typename T, typename ... Others>
	void Print(T const &v, Others const & ... others)
	{
		std::cout << v << " ";
		Print(std::forward<Others const &>(others )...);
	}

};

int main(int argc, char **argv)
{
	AcceptorBase * f1 = dynamic_cast<AcceptorBase*>(new Foo1);
	auto v1 = CreateVisitor<Foo1>("Command1", 5, 6);
	auto v2 = CreateVisitor<Foo1>("Command2", "hello world");
	auto v3 = CreateVisitor<Foo1>("Command3", 5, 6, 3);
	f1->accept(v1);
	f1->accept(v2);
	f1->accept(v3);

	delete f1;

}

