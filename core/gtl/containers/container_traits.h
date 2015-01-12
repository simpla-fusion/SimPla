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

#include "../design_pattern/memory_pool.h"

namespace simpla
{

/** @ingroup container
 * @{
 */

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

	static auto get_value(container_type & d, size_t s)
	DECL_RET_TYPE ((d[s]))
	static auto get_value(container_type const& d, size_t s)
	DECL_RET_TYPE ((d[s]))

};

template<typename TV> struct container_traits<std::shared_ptr<TV>>
{
	typedef std::shared_ptr<TV> container_type;
	typedef TV value_type;

	static container_type allocate(size_t s)
	{
		return sp_make_shared_array<value_type>(s);
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
	static TV & get_value(container_type & d, size_t s)
	{
		return d.get()[s];
	}
	static TV const & get_value(container_type const& d, size_t s)
	{
		return d.get()[s];
	}
};
/** @}*/
}
// namespace simpla
#endif /* CONTAINER_TRAITS_H_ */
