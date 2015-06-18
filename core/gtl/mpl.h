/*
 * mpl.h
 *
 *  Created on: 2015年6月12日
 *      Author: salmon
 */

#ifndef CORE_GTL_MPL_H_
#define CORE_GTL_MPL_H_

namespace simpla
{

namespace mpl
{
template<size_t N, typename _TP, _TP ...I> struct unpack_int_seq;

template<typename _Tp, _Tp I0, _Tp ...I>
struct unpack_int_seq<0, _Tp, I0, I...> : public std::integral_constant<_Tp, I0>
{

};
template<size_t N, typename _Tp, _Tp I0, _Tp ...I>
struct unpack_int_seq<N, _Tp, I0, I...> : public std::integral_constant<_Tp,
		unpack_int_seq<N - 1, _Tp, I...>::value>
{
};

template<size_t N, typename _Tp>
struct unpack_int_seq<N, _Tp> : public std::integral_constant<_Tp, 0>
{
};

template<unsigned int, typename ...> struct unpack_type_seq;
template<unsigned int N, typename ...T>
using unpack_type_seq_t=typename unpack_type_seq<N,T...>::type;

template<typename T0, typename ...Others>
struct unpack_type_seq<0, T0, Others...>
{
	typedef T0 type;
};
template<unsigned int N>
struct unpack_type_seq<N>
{
	typedef void type;
};
template<unsigned int N, typename T0, typename ...Others>
struct unpack_type_seq<N, T0, Others...>
{
	typedef typename unpack_type_seq<N - 1, Others...>::type type;
};

template<typename, typename ...> struct find_type_in_list;
template<typename T, typename ...Others>
using find_type_in_list_t=typename find_type_in_list<T,Others...>::type;

template<typename T>
struct find_type_in_list<T>
{
	static constexpr bool value = false;
};
template<typename T, typename U>
struct find_type_in_list<T, U>
{
	static constexpr bool value = std::is_same<T, U>::value;

	typedef typename std::conditional<value, T, void>::type type;
};
template<typename T, typename U, typename ...Others>
struct find_type_in_list<T, U, Others...>
{
	static constexpr bool value = find_type_in_list<T, U>::value
			|| find_type_in_list<T, Others...>::value;

	typedef typename std::conditional<value, T, void>::type type;

};

template<typename T, typename ...Others>
using find_type_in_list_t=typename find_type_in_list<T, Others...>::type;

template<typename _Tp, _Tp ... Others> struct max;
template<typename _Tp, _Tp first, _Tp second>
struct max<_Tp, first, second> : std::integral_constant<_Tp,
		(first > second) ? first : second>
{
};
template<typename _Tp, _Tp first, _Tp ... Others>
struct max<_Tp, first, Others...> : std::integral_constant<_Tp,
		max<_Tp, first, max<_Tp, Others...>::value>::value>
{
};

template<typename _Tp, _Tp ... Others> struct min;
template<typename _Tp, _Tp first, _Tp second>
struct min<_Tp, first, second> : std::integral_constant<_Tp,
		(first > second) ? first : second>
{
};
template<typename _Tp, _Tp first, _Tp ... Others>
struct min<_Tp, first, Others...> : std::integral_constant<_Tp,
		min<_Tp, first, min<_Tp, Others...>::value>::value>
{
};

}
// namespace mpl

}// namespace simpla

#endif /* CORE_GTL_MPL_H_ */
