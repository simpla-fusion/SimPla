/*
 * test1.cpp
 *
 *  Created on: 2013年12月11日
 *      Author: salmon
 */

#include <ios>
#include <iostream>
#include <type_traits>
#include "../src/utilities/log.h"
#include "../src/utilities/type_utilites.h"

HAS_MEMBER_FUNCTION(size);
HAS_OPERATOR(index, []);
struct Foo
{

	void size(int);
	void operator[](int);

};

struct Foo2
{
	void size();
};

int main()
{
	CHECK((has_member_function_size<Foo, double>::value));
	CHECK((has_member_function_size<Foo, Foo>::value));
	CHECK((has_operator_index<Foo, int>::value));
	CHECK((has_operator_index<Foo2, int>::value));

}
