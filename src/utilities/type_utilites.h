/*
 * type_utilites.h
 *
 *  Created on: 2013年12月14日
 *      Author: salmon
 */

#ifndef TYPE_UTILITES_H_
#define TYPE_UTILITES_H_

#include "log.h"

namespace simpla
{

#include <tuple>

template<bool N> struct Bool2Type
{
	static const bool value = N;
};

template<int N> struct Int2Type
{
	static const int value = N;
};

struct NullType;

struct EmptyType
{
};

enum CONST_NUMBER
{
	ZERO = 0, ONE = 1, TWO = 2, THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8, NINE = 9
};

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
	static constexpr bool value = !std::is_same<decltype(0), no>::value;                     \
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

#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}

#define ENABLE_IF_DECL_RET_TYPE(_COND_,_EXPR_) \
        ->typename std::enable_if<_COND_,decltype((_EXPR_))>::type {return (_EXPR_);}

#define COND_DECL_RET_TYPE(_COND_,_EXPR_,_FAILSAFE_) \
        ->typename std::conditional<_COND_,decltype((_EXPR_)),_FAILSAFE_>::type {return (_EXPR_);}

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

template<class T, typename TI = int>
class is_indexable
{
//	HAS_OPERATOR(index, []);
public:
	static const bool value = false; // has_operator_index<T, TI>::value;

};
template<class T>
class is_indexable<T*>
{
public:
	static const bool value = true;

};
// @ref http://stackoverflow.com/questions/3913503/metaprogram-for-bit-counting
template<int N>
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
	while (s > 0)
	{
		++n;
		s = s >> 1;
	}
	return n;
}

//******************************************************************************************************
// iterator
//******************************************************************************************************

template<typename TI>
struct Range
{
	typedef TI iterator;
	iterator ib, ie;
	Range()
	{
	}

	template<typename T>
	Range(T b, T e)
			: ib(b), ie(e)
	{
	}

	~Range()
	{
	}
	iterator begin() const
	{
		return ib;
	}
	iterator end() const
	{
		return ie;
	}

	Range<TI> split(int num, int id)
	{
		return Range<TI>(ib + (ie - ib) * id / num, ib + (ie - ib) * (id + 1) / num);
	}
};


template<typename T> inline T* PointerTo(T & v)
{
	return &v;
}

template<typename T> inline T* PointerTo(T * v)
{
	return v;
}

} // namespace simpla

#endif /* TYPE_UTILITES_H_ */
