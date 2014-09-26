/*
 * integer_seq.cpp
 *
 *  Created on: 2014年9月26日
 *      Author: salmon
 */
#include <iostream>
#include <typeinfo>
template<typename _Tp, _Tp ... _Idx>
struct integer_sequence

{
	typedef _Tp value_type;

	static constexpr _Tp data[sizeof...(_Idx)] =
			{ _Idx... };

			static constexpr size_t size()
			{
				return (sizeof...(_Idx));
					}

				};

				template<typename ...> class cat_integer_sequence;

template<typename T, T ... N1, T ... N2>
struct cat_integer_sequence<integer_sequence<T, N1...>,
		integer_sequence<T, N2...>>
{
	typedef integer_sequence<T, N1..., N2...> type;
};

//template<typename ... >class show;
//template<typename T, T M, T ... N1>
//struct show<integer_sequence<T, M, N1...>>
//{
//	static void eval()
//	{
//		std::cout << M << " ";
//		show<integer_sequence<T, N1...>>::eval();
//	}
//};
//template<typename T>
//struct show<integer_sequence<T>>
//{
//	static void eval()
//	{
//		std::cout << " END " << std::endl;
//	}
//};

template<typename ...> struct make_array_type;
template<typename T, unsigned int M, unsigned int ...N>
struct make_array_type<T, integer_sequence<unsigned int, M, N...> >
{
	typedef typename make_array_type<T, integer_sequence<unsigned int, N...>>::type sub_type;

	typedef sub_type type[M];
};

template<typename T, unsigned int M>
struct make_array_type<T, integer_sequence<unsigned int, M> >
{
	typedef T type[M];
};

int main(int argc, char **argv)
{
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

	typedef integer_sequence<unsigned int, 3, 4> seq;
	typedef typename make_array_type<double, seq>::type t0;
	typedef double t1[3][4];
	typedef double t2[4][3];

	std::cout << std::boolalpha << std::is_same<t0, t1>::value << std::endl;
	std::cout << std::boolalpha << std::is_same<t0, t2>::value << std::endl;
	std::cout << typeid(t0).name() << std::endl;
	std::cout << typeid(t1).name() << std::endl;
	std::cout << typeid(t2).name() << std::endl;

//	std::cout << get_seq<0, seq>::value << std::endl;
//	std::cout << get_seq<1, seq>::value << std::endl;
//	std::cout << get_seq<2, seq>::value << std::endl;
//	std::cout << get_seq<3, seq>::value << std::endl;
//	std::cout << get_seq<4, seq>::value << std::endl;

}
