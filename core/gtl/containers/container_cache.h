/**
 * @file container_cache.h
 *
 *  created on: 2014-7-1
 *      Author: salmon
 */

#ifndef CONTAINER_CACHE_H_
#define CONTAINER_CACHE_H_

#include "../type_traits.h"
namespace simpla
{

/**
 * \brief  Cache is transparent, and would not change the behavior
 * of source object. In default, Cache do nothing. It only affect
 * the object by the template specialization of object.
 *
 *
 *
 */
template<typename T>
struct CacheContainer
{
	T & f_;

	template<typename ... Args>
	CacheContainer(T& f, Args ...)
			: f_(f)
	{

	}

	T & operator*()
	{
		return f_;
	}
};

template<typename T>
struct CacheContainer<T const &>
{
	T const & f_;

	template<typename ... Args>
	CacheContainer(T const & f, Args ...)
			: f_(f)
	{

	}

	T const & operator*() const
	{
		return f_;
	}
};

template<typename T>
struct CacheContainer<T*>
{
	T * f_;

	template<typename ... Args>
	CacheContainer(T* f, Args ...)
			: f_(f)
	{
	}

	T & operator*()
	{
		return *f_;
	}
};

template<typename TIndexType, typename T>
void RefreshCache(TIndexType s, T &)
{
}
template<typename TIndexType, typename T>
void RefreshCache(TIndexType s, T *)
{
}
template<typename TIndexType, typename T>
void RefreshCache(TIndexType s, CacheContainer<T> & c)
{
	RefreshCache(s, *c);
}

template<typename TIndexType, typename T, typename ...Others>
void RefreshCache(TIndexType s, T & f, Others & ...others)
{
	RefreshCache(s, f);
	RefreshCache(s, others...);
}

template<typename T>
void FlushCache(T &)
{
}
template<typename T>
void FlushCache(T *f)
{
	FlushCache(*f);
}
template<typename T, typename ...Others>
void FlushCache(T & f, Others & ...others)
{
	FlushCache(f);
	FlushCache(others...);
}

template<typename T> auto make_cache(T&& v)
DECL_RET_TYPE((CacheContariner<T>(std::forward<T>(v))))
}  // namespace simpla

#endif /* CONTAINER_CACHE_H_ */
