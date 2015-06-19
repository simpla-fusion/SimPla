/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */
#include <iostream>
#include <typeinfo>
#include <tuple>
#include "../../core/gtl/mpl.h"

using namespace simpla;
template<typename ...> struct Foo
{

};
int main()
{
	typedef std::tuple<int, double, bool> t1;
	typedef std::tuple<int, bool, bool> t2;

	std::cout << std::boolalpha << std::endl

	<< std::is_same<mpl::replace_tuple_t<1, bool, t1>, t2>::value << std::endl

	<< " : " << typeid(t1).name() << std::endl

	<< " : " << typeid(mpl::replace_tuple_t<0, bool, t1>).name() << std::endl

	<< " : " << typeid(mpl::replace_tuple_t<1, bool, t1>).name() << std::endl

	<< " : " << typeid(mpl::replace_tuple_t<2, bool, t1>).name() << std::endl

	<< " : " << typeid(t2).name() << std::endl

	<< " : " << typeid(mpl::assamble_tuple_t<Foo, int, double>).name()
			<< std::endl

			<< " : " << typeid(mpl::assamble_tuple_t<Foo, t1>).name()
			<< std::endl;
}
