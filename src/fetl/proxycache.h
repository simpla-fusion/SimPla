/*
 * proxycache.h
 *
 *  Created on: 2013年11月12日
 *      Author: salmon
 */

#ifndef PROXYCACHE_H_
#define PROXYCACHE_H_
#include "../utilities/type_utilites.h"
namespace simpla
{

/**
 * @brief  ProxyCache is transparent, and would not change the behavior
 * of source object. In default, ProxyCache do nothing. It only affect
 * the object by the template specialization of object.
 *
 * for Example:
 *
 template<typename TGeometry, typename TValue>
 struct ProxyCache<const Field<TGeometry, TValue> >
 {
 typedef const Field<TGeometry, TValue> src_type;

 typedef Field<TGeometry, ProxyCache<src_type> > type;

 template<typename TI>
 static inline type Eval(src_type & f, TI const &hint_idx)
 {
 return std::move(type(f, hint_idx));
 }

 };
 *
 * where Field<TGeometry, ProxyCache<src_type> > is the cached version
 * Field.
 *
 * @ref: fetl/proxycache.h
 */
template<typename T>
struct ProxyCache
{
	typedef T &reference;
	template<typename ...TI>
	static inline T Eval(T &f, TI ...)
	{
		return std::forward<T>(f);
	}
};

template<typename T>
struct ProxyCache<T*>
{
	typedef T *reference;
	template<typename ...TI>
	static inline T* Eval(T *f, TI ...)
	{
		return f;
	}
};

template<typename T>
void RefreshCache(size_t s, T &)
{
}

template<typename T, typename ...Others>
void RefreshCache(size_t s, T & f, Others & ...others)
{
	RefreshCache(s, f);
	RefreshCache(s, others...);
}

template<typename T>
void FlushCache(T &)
{
}

template<typename T, typename ...Others>
void FlushCache(T & f, Others & ...others)
{
	FlushCache(f);
	FlushCache(others...);
}

}  // namespace simpla

#endif /* PROXYCACHE_H_ */
