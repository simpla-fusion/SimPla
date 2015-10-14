/*
 * field_test.cpp
 *
 *  created on: 2014-6-30
 *      Author: salmon
 */


#include <iostream>
#include <vector>
#include <typeinfo>

#include "field.h"

using namespace simpla;


struct Base
{
	typedef double type;
};
struct Derived : public Base
{

};

int main(int argc, char **argv)
{
	Field<double> a;
	std::cout << a.size() << std::endl;
	std::cout << typeid(Derived::type).name() << std::endl;
}
