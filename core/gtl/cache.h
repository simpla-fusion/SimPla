/**
 * @file cache.h
 *
 *  created on: 2013-11-12
 *      Author: salmon
 */

#ifndef CORE_GTL_CACHE_H_
#define CORE_GTL_CACHE_H_
#include "type_traits.h"
namespace simpla
{

/**
 * @ingroup gtl
 *
 * \brief  CellCache is transparent, and would not change the behavior
 * of source object. In default, CellCache do nothing. It only affect
 * the object by the template specialization of object.
 *
 *
 */
template<typename T>
struct Cache
{
	typedef T cached_type;

	T & f_;

	template<typename ... Args>
	Cache(T& f, Args ...) :
			f_(f)
	{
	}

	T & operator*()
	{
		return f_;
	}

};

template<typename T>
struct Cache<T const &>
{

	typedef T const & cached_type;

	T const & f_;

	template<typename ... Args>
	Cache(T const & f, Args ...) :
			f_(f)
	{
	}

	T const & operator*() const
	{
		return f_;
	}

};

template<typename T>
struct Cache<T*>
{
	typedef T * cached_type;
	typedef Cache<T*> this_type;

	T * f_;

	template<typename ... Args>
	Cache(T* f, Args ...) :
			f_(f)
	{
	}

	T & operator*()
	{
		return *f_;
	}

};

template<typename T>
constexpr bool is_cached(Cache<T> &&)
{
	return true;
}

template<typename T, typename ...Args>
bool is_cached(T && first, Args && ...args)
{
	return std::is_same<T, typename Cache<T>::cached_type>::value
			&& is_cached(std::forward<Args>(args)...);
}

template<typename T, typename ...Args>
typename Cache<T>::cached_type cache(Args&& ...args)
{
	return Cache<T>::create(std::forward<Args>(args)...);
}

/**
 *  @brief  Forward an lvalue.
 *  @return The parameter cast to the specified type.
 *
 *  This function is used to implement "cache forwarding".
 */
template<typename _Tp>
constexpr _Tp&&
cache_forward(_Tp & __t) noexcept
{
	return std::forward<_Tp>(__t);
}

//template<typename TIndexType, typename T>
//void RefreshCache(TIndexType s, T &)
//{
//}
//template<typename TIndexType, typename T>
//void RefreshCache(TIndexType s, T *)
//{
//}
//template<typename TIndexType, typename T>
//void RefreshCache(TIndexType s, CellCache<T> & c)
//{
//	RefreshCache(s, *c);
//}
//
//template<typename TIndexType, typename T, typename ...Others>
//void RefreshCache(TIndexType s, T & f, Others & ...others)
//{
//	RefreshCache(s, f);
//	RefreshCache(s, others...);
//}
//
//template<typename T>
//void FlushCache(T &)
//{
//}
//template<typename T>
//void FlushCache(T *f)
//{
//	FlushCache(*f);
//}
//template<typename T, typename ...Others>
//void FlushCache(T & f, Others & ...others)
//{
//	FlushCache(f);
//	FlushCache(others...);
//}

}// namespace simpla

#endif /* CORE_GTL_CACHE_H_ */
