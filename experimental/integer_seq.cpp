/*
 * integer_seq.cpp
 *
 *  Created on: 2014年9月26日
 *      Author: salmon
 */
#include <iostream>
#include "../src/utilities/sp_integer_sequence.h"
#include "../src/utilities/ntuple.h"
#include "../src/utilities/pretty_stream.h"
//struct FOO
//{
//	void show() const
//	{
//		std::cout << std::endl;
//	}
//	template<typename T1, typename ...T>
//	void show(T1 const &u, T const & ...v) const
//	{
//		std::cout << u << " ";
//		show(v...);
//	}
//	template<typename ...T>
//	void operator()(T ...args) const
//	{
//		show(args...);
//	}
//
//};

using namespace simpla;
int main(int argc, char **argv)
{

//	seq_for<integer_sequence<unsigned int, 1, 2, 3, 4>>::eval_multi_parameter(
//			FOO(), "hello", "world");

	std::cout << integer_sequence<unsigned int, 1, 2, 3, 4>::value()
			<< std::endl;
//	show<
//			typename cat_integer_sequence<
//					integer_sequence<unsigned int, 1, 3, 4, 7, 9>,
//					integer_sequence<unsigned int, 2, 4, 6, 8, 10>>::type>::eval();
//
//	show<
//			typename cat_integer_sequence<integer_sequence<unsigned int>,
//					integer_sequence<unsigned int, 2, 4, 6, 8, 10>>::type>::eval();
//
//	show<
//			typename cat_integer_sequence<
//					integer_sequence<unsigned int, 1, 3, 4, 7, 9>,
//					integer_sequence<unsigned int>>::type>::eval();

//	typedef integer_sequence<unsigned int, 3, 4> seq;
//	typedef typename make_array_type<double, seq>::type t0;
//	typedef double t1[3][4];
//	typedef double t2[4][3];
//
//	std::cout << std::boolalpha << std::is_same<t0, t1>::value << std::endl;
//	std::cout << std::boolalpha << std::is_same<t0, t2>::value << std::endl;
//	std::cout << typeid(t0).name() << std::endl;
//	std::cout << typeid(t1).name() << std::endl;
//	std::cout << typeid(t2).name() << std::endl;

//	std::cout << get_seq<0, seq>::value << std::endl;
//	std::cout << get_seq<1, seq>::value << std::endl;
//	std::cout << get_seq<2, seq>::value << std::endl;
//	std::cout << get_seq<3, seq>::value << std::endl;
//	std::cout << get_seq<4, seq>::value << std::endl;

}
