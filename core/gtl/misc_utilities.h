/*
 * misc_utilities.h
 *
 *  Created on: 2015年6月11日
 *      Author: salmon
 */

#ifndef CORE_GTL_MISC_UTILITIES_H_
#define CORE_GTL_MISC_UTILITIES_H_
#include "check_concept.h"
namespace simpla
{
/**
 * Count the number of arguments passed to MACRO, very carefully
 * tiptoeing around an MSVC bug where it improperly expands __VA_ARGS__ as a
 * single token in argument lists.  See these URLs for details:
 *
 *   http://stackoverflow.com/questions/9183993/msvc-variadic-macro-expansion/9338429#9338429
 *   http://connect.microsoft.com/VisualStudio/feedback/details/380090/variadic-macro-replacement
 *   http://cplusplus.co.il/2010/07/17/variadic-macro-to-count-number-of-arguments/#comment-644
 */
#define COUNT_MACRO_ARGS_IMPL2(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18, count, ...) count
#define COUNT_MACRO_ARGS_IMPL(args) COUNT_MACRO_ARGS_IMPL2 args
#define COUNT_MACRO_ARGS(...) COUNT_MACRO_ARGS_IMPL((__VA_ARGS__,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))

/// \note  http://stackoverflow.com/questions/3913503/metaprogram-for-bit-counting
template<unsigned long N> struct CountBits
{
	static const unsigned long n = CountBits<N / 2>::n + 1;
};

template<> struct CountBits<0>
{
	static const unsigned long n = 0;
};

inline unsigned long count_bits(unsigned long s)
{
	unsigned long n = 0;
	while (s != 0)
	{
		++n;
		s = s >> 1;
	}
	return n;
}

template<typename T>
T const & min(T const & first, T const & second)
{
	return std::min(first, second);
}

template<typename T>
T const & min(T const & first)
{
	return first;
}
template<typename T, typename ...Others>
T const & min(T const & first, Others &&... others)
{
	return min(first, min(std::forward<Others>(others)...));
}

template<typename T>
T const & max(T const & first, T const & second)
{
	return std::max(first, second);
}

template<typename T>
T const & max(T const & first)
{
	return first;
}

template<typename T, typename ...Others>
T const & max(T const & first, Others &&...others)
{
	return max(first, max(std::forward<Others>(others)...));
}
HAS_MEMBER_FUNCTION(swap)

template<typename T> typename std::enable_if<has_member_function_swap<T>::value,
		void>::type sp_swap(T& l, T& r)
{
	l.swap(r);
}

template<typename T> typename std::enable_if<
		!has_member_function_swap<T>::value, void>::type sp_swap(T& l, T& r)
{
	std::swap(l, r);
}

//template<typename TV>
//auto print(std::ostream & os,
//		TV const & v)
//		->typename std::enable_if<!has_member_function_print<TV const,std::ostream &>::value,std::ostream &>::type
//{
//	os << v;
//	return os;
//}
}  // namespace simpla

#endif /* CORE_GTL_MISC_UTILITIES_H_ */
