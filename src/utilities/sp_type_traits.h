/*
 * sp_type_traits.h
 *
 *  created on: 2014-6-15
 *      Author: salmon
 */

#ifndef SP_TYPE_TRAITS_H_
#define SP_TYPE_TRAITS_H_

#include <type_traits>
#include <memory>
#include <tuple>

namespace simpla
{

template<bool B> using Bool2Type=std::integral_constant<bool,B>;

template<unsigned int B> using Int2Type=std::integral_constant<unsigned int ,B>;

typedef std::nullptr_t NullType;

struct EmptyType
{
};

enum CONST_NUMBER
{
	ZERO = 0, ONE = 1, TWO = 2, THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8, NINE = 9
};

#define HAS_MEMBER(_NAME_)                                                                 \
template<typename _T>                                                                      \
struct has_member_##_NAME_                                                                 \
{                                                                                          \
private:                                                                                   \
	typedef std::true_type yes;                                                            \
	typedef std::false_type no;                                                            \
                                                                                           \
	template<typename U>                                                                   \
	static auto test(int) ->  decltype(std::declval<U>()._NAME_  )                         \
	template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
	static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;        \
};

#define HAS_TYPE_MEMBER(_NAME_)                                                            \
template<typename _T,typename _D>                                                          \
struct has_type_member_##_NAME_                                                            \
{                                                                                          \
private:                                                                                   \
	typedef std::true_type yes;                                                            \
	typedef std::false_type no;                                                            \
                                                                                           \
	template<typename U>                                                                   \
	static auto test(int) ->  U::_NAME_                        \
	template<typename > static no test(...);                                               \
                                                                                           \
public:                                                                                    \
	static constexpr bool value = std::is_same<decltype(test< _T>(0)), _D>::value;         \
};                                                                                         \
                                                                                           \
template<typename _T, typename _D>                                                         \
typename std::enable_if<has_type_member_##_NAME_<_T, _D>::value, _D>::type                 \
get_member_##_NAME_(_T const & c, _D const & def){	return c._NAME_; }                     \
template<typename _T, typename _D>                                                         \
typename std::enable_if<!has_type_member_##_NAME_<_T, _D>::value, _D>::type                \
get_member_##_NAME_(_T const & c, _D const & def){	return def;}                           \


#define HAS_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(std::declval<U>()._NAME_() )>::type;                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(std::declval<U>()._NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
};

#define HAS_CONST_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_const_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(std::declval<const U>()._NAME_() )>::type;                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(std::declval<const U>()._NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
};

#define HAS_STATIC_MEMBER_FUNCTION(_NAME_)                                                                   \
template<typename _T, typename ..._Args>                                                                \
struct has_static_member_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(U::_NAME_() )>::type;                                                       \
                                                                                                      \
	template<typename U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(U::_NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test< _T>(0)), no>::value;                     \
};

#define HAS_FUNCTION(_NAME_)                                                                   \
template< typename ..._Args>                                                                \
struct has_function_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(_NAME_() )>::type;                                                       \
                                                                                                      \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(_NAME_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test(0)), no>::value;                     \
};

#define HAS_OPERATOR(_NAME_,_OP_)                                                                   \
template<typename _T, typename ... _Args>                                                                \
struct has_operator_##_NAME_                                                                    \
{                                                                                                     \
private:                                                                                              \
	typedef std::true_type yes;                                                                       \
	typedef std::false_type no;                                                                       \
                                                                                                      \
	template<typename _U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< sizeof...(_Args)==0,                                                      \
	decltype(std::declval<_U>().operator _OP_() )>::type;                                                       \
                                                                                                      \
	template<typename _U>                                                                              \
	static auto test(int) ->                                                                          \
	typename std::enable_if< ( sizeof...(_Args) >0),                                                   \
	decltype(std::declval<_U>().operator _OP_(std::declval<_Args>()...) )>::type;                            \
                                                                                                      \
	template<typename > static no test(...);                                                          \
                                                                                                      \
public:                                                                                               \
                                                                                                      \
	static constexpr bool value = !std::is_same<decltype(test<_T>(0)), no>::value;                     \
};

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return std::move(_EXPR_);}

#define ENABLE_IF_DECL_RET_TYPE(_COND_,_EXPR_) \
        ->typename std::enable_if<_COND_,decltype((_EXPR_))>::type {return (_EXPR_);}

//template<typename T>
//struct remove_const_reference
//{
//	typedef typename std::remove_const<typename std::remove_reference<T>::type>::type type;
//};
//template<typename T>
//struct is_storage_type
//{
//	static constexpr bool value = true;
//};
//template<typename T>
//struct is_storage_type<T*>
//{
//	static constexpr bool value = true;
//};
//template<typename T>
//struct ReferenceTraits // obsolete
//{
//	typedef typename remove_const_reference<T>::type TL;
//	typedef typename std::conditional<is_storage_type<TL>::value, TL &, TL>::type type;
//};
//
//template<typename T>
//struct ConstReferenceTraits // obsolete
//{
//	typedef typename remove_const_reference<T>::type TL;
//	typedef typename std::conditional<is_storage_type<TL>::value, TL const &, const TL>::type type;
//};

template<typename T>
struct can_not_reference
{
	static constexpr bool value = false;
};

template<typename T>
struct StorageTraits
{
	static constexpr bool not_refercne = std::is_pointer<T>::value

	|| std::is_reference<T>::value

	|| std::is_scalar<T>::value

	|| can_not_reference<T>::value;

	typedef typename std::conditional<not_refercne, T, T&>::type type;

	typedef typename std::conditional<not_refercne, T, const T&>::type const_reference;

	typedef typename std::conditional<not_refercne, T, T&>::type reference;

};

HAS_OPERATOR(sub_script, []);
HAS_MEMBER_FUNCTION(at);
template<class T, typename TI = int>
class is_indexable
{
public:
	static const bool value = has_operator_sub_script<T, TI>::value; // has_operator_index<T, TI>::value;
};
template<class T>
class is_indexable<T*, size_t>
{
public:
	static const bool value = true;
};
template<class T>
class is_indexable<T*, int>
{
public:
	static const bool value = true;
};

template<typename T, typename TI>
auto get_value(T & v, TI const & s)
ENABLE_IF_DECL_RET_TYPE((is_indexable<T,TI>::value), v[s])
;

template<typename T, typename TI>
auto get_value(T & v, TI const & s)
ENABLE_IF_DECL_RET_TYPE(((!is_indexable<T,TI>::value) && has_member_function_at<T,TI>::value), v.at(s))
;

template<typename T, typename TI>
auto get_value(T & v, TI const & s)
ENABLE_IF_DECL_RET_TYPE((!(is_indexable<T,TI>::value || has_member_function_at<T,TI>::value)),(v))
;

template<typename T, typename TI>
auto get_value(std::shared_ptr<T> v, TI const & s)
DECL_RET_TYPE(get_value(*v ,s))
;

/// \note  http://stackoverflow.com/questions/3913503/metaprogram-for-bit-counting
template<unsigned int N>
struct CountBits
{
	static const unsigned int n = CountBits<N / 2>::n + 1;
};

template<>
struct CountBits<0>
{
	static const unsigned int n = 0;
};

inline unsigned int count_bits(unsigned long s)
{
	unsigned int n = 0;
	while (s != 0)
	{
		++n;
		s = s >> 1;
	}
	return n;
}

template<typename T> inline T* PointerTo(T & v)
{
	return &v;
}

template<typename T> inline T* PointerTo(T * v)
{
	return v;
}

template<typename TV, typename TR> inline TV TypeCast(TR const& obj)
{
	return std::move(static_cast<TV>(obj));
}

//namespace _impl
//{
////******************************************************************************************************
//// Third-part code begin
//// ref: https://gitorious.org/redistd/redistd
//// Copyright Jonathan Wakely 2012
//// Distributed under the Boost Software License, Version 1.0.
//// (See accompanying file LICENSE_1_0.txt or copy at
//// http://www.boost.org/LICENSE_1_0.txt)
//
///// A type that represents a parameter pack of zero or more integers.
//template<unsigned ... Indices>
//struct index_tuple
//{
//	/// Generate an index_tuple with an additional element.
//	template<unsigned N>
//	using append = index_tuple<Indices..., N>;
//};
//
///// Unary metafunction that generates an index_tuple containing [0, Size)
//template<unsigned Size>
//struct make_index_tuple
//{
//	typedef typename make_index_tuple<Size - 1>::type::template append<Size - 1> type;
//};
//
//// Terminal case of the recursive metafunction.
//template<>
//struct make_index_tuple<0u>
//{
//	typedef index_tuple<> type;
//};
//
//template<typename ... Types>
//using to_index_tuple = typename make_index_tuple<sizeof...(Types)>::type;
//// Third-part code end
////******************************************************************************************************
//
//}// namespace _impl
}// namespace simpla
namespace std
{
template<typename T>
T begin(std::pair<T, T>const & range)
{
	return std::move(range.first);
}
template<typename T>
T end(std::pair<T, T>const & range)
{
	return std::move(range.second);
}
template<typename T>
T rbegin(std::pair<T, T>const & range)
{
	return std::move(range.second--);
}
template<typename T>
T rend(std::pair<T, T>const & range)
{
	return std::move(range.first--);
}

template<typename T> T const &Compact(T const &v)
{
	return v;
}

template<typename T> void Decompact(T const &v, T * u)
{
	*u = v;
}
}
#endif /* SP_TYPE_TRAITS_H_ */
