/*
 * test2.cpp
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

//#include <cstring>

#include <iostream>
#include <string>
#include <type_traits>

template<class T, typename TI = int>
struct is_indexable
{
	template<typename T1, typename T2>
	static auto check_index(T1 const& u,
			T2 const &s) -> decltype(const_cast<typename std::remove_cv<T1>::type &>(u)[s])
	{
	}

	template<typename T1, typename T2>
	static auto check_const_index_only(T1 const &u,
			T2 const &s) -> decltype(u[s])
	{
	}

	static std::false_type check_index(...)
	{
		return std::false_type();
	}
	static std::false_type check_const_index_only(...)
	{
		return std::false_type();
	}
public:

//	typedef decltype(
//			check_const_index(std::declval<T>(),
//					std::declval<TI>())) const_result_type;

	typedef decltype(
			check_index((std::declval<T>()),
					std::declval<TI>())) result_type;

	typedef decltype(
			check_const_index_only((std::declval<T>()),
					std::declval<TI>())) const_result_type;

	static const bool value =
			!(std::is_same<result_type, std::false_type>::value);

	static const bool has_const_ref = !(std::is_same<const_result_type,
			std::false_type>::value);

	static const bool has_non_const_ref = value
			&& (!std::is_const<result_type>::value);

//			!(std::is_same<const_result_type, std::false_type>::value)
	;
};
template<typename T, typename TI> inline typename std::enable_if<
		!is_indexable<T, TI>::value, T>::type index(T const & v, TI const &)
{
	return (v);
}

template<typename T, typename TI> inline typename std::enable_if<
		is_indexable<T, TI>::value, typename is_indexable<T, TI>::result_type>::type index(
		T const & v, TI const &s)
{
	return (v[s]);
}
int main()
{
	double a = 4;
	std::cout << index(a, 1) << std::endl;
	std::cout << std::boolalpha << is_indexable<decltype(a), decltype(1)>::value
			<< std::endl;

}

