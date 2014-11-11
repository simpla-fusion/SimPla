/*
 * container_traits.h
 *
 *  Created on: Oct 13, 2014
 *      Author: salmon
 */

#ifndef CONTAINER_TRAITS_H_
#define CONTAINER_TRAITS_H_

#include <memory>
#include <string.h>

#include "../utilities/memory_pool.h"

namespace simpla
{

template<typename TContainer> struct container_traits
{
	typedef TContainer container_type;
	typedef typename TContainer::value_type value_type;

	static container_type allocate(size_t s)
	{
		return std::move(container_type(s));
	}
	template<typename ...T>
	static void clear(T &&...)
	{
	}

	static bool is_empty(container_type const& that)
	{
		return that.empty();
	}

	static bool is_same(container_type const& lhs, container_type const& rhs)
	{
		return lhs == rhs;
	}
};

template<typename TV> struct container_traits<std::shared_ptr<TV>>
{
	typedef std::shared_ptr<TV> container_type;
	typedef TV value_type;

	static container_type allocate(size_t s)
	{
		return std::shared_ptr<value_type>(new value_type[s],
				deallocate_m<value_type>);
//		return MEMPOOL.template make_shared < value_type > (s);
	}

	static void clear(std::shared_ptr<TV> d, size_t s)
	{
		memset(d.get(), 0, s * sizeof(TV));
	}
	static bool is_empty(container_type const& that)
	{
		return that == nullptr;
	}

	static bool is_same(container_type const& lhs, container_type const& rhs)
	{
		return lhs == rhs;
	}
};

}
// namespace simpla
#endif /* CONTAINER_TRAITS_H_ */
