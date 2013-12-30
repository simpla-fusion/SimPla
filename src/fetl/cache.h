/*
 * cache.h
 *
 *  Created on: 2013年11月12日
 *      Author: salmon
 */

#ifndef CACHE_H_
#define CACHE_H_
#include "../utilities/type_utilites.h"
namespace simpla
{

/**
 * @brief  Cache is transparent, and would not change the behavior
 * of source object. In default, Cache do nothing. It only affect
 * the object by the template specialization of object.
 *
 *
 *
 */
template<typename T>
struct Cache
{
	T & f_;

	template<typename ... Args>
	Cache(T& f, Args ...)
			: f_(f)
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
	T const & f_;

	template<typename ... Args>
	Cache(T const & f, Args ...)
			: f_(f)
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
	T * f_;

	template<typename ... Args>
	Cache(T* f, Args ...)
			: f_(f)
	{
	}

	T & operator*()
	{
		return *f_;
	}
};

template<typename T>
void RefreshCache(size_t s, T &)
{
}
template<typename T>
void RefreshCache(size_t s, T *)
{
}
template<typename T>
void RefreshCache(size_t s, Cache<T> & c)
{
	RefreshCache(s, *c);
}

template<typename T, typename ...Others>
void RefreshCache(size_t s, T & f, Others & ...others)
{
	RefreshCache(s, f);
	RefreshCache(s, others...);
}

template<typename T>
void FlushCache(Cache<T> & c)
{
	FlushCache(*c);
}

template<typename T>
void FlushCache(T &)
{
}
template<typename T>
void FlushCache(T *)
{
}
template<typename T, typename ...Others>
void FlushCache(T & f, Others & ...others)
{
	FlushCache(f);
	FlushCache(others...);
}

}  // namespace simpla

#endif /* CACHE_H_ */
