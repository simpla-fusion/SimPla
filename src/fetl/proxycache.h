/*
 * proxycache.h
 *
 *  Created on: 2013年11月12日
 *      Author: salmon
 */

#ifndef PROXYCACHE_H_
#define PROXYCACHE_H_
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
	typedef T src_type;
	typedef T type;
	template<typename TI>
	static inline src_type & Eval(src_type & f, TI const &)
	{
		return f;
	}
};

}  // namespace simpla

#endif /* PROXYCACHE_H_ */
