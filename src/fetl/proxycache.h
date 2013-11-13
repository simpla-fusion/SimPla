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
