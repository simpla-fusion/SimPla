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

#include "../../utilities/memory_pool.h"
#include "../type_traits.h"
namespace simpla
{

/** @ingroup container
 * @{
 */

template<typename TContainer>
struct container_traits
{
	typedef TContainer container_type;
	typedef typename TContainer::key_type key_type;
	typedef typename TContainer::mapped_type value_type;
	typedef std::shared_ptr<container_type> holder_type;
//	HAS_TYPE(key_type);
	static constexpr bool is_associative_container = true;

	static holder_type allocate(size_t s)
	{
		return std::move(std::make_shared<container_type>(s));
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

	static value_type & get_value(holder_type & data, key_type s)
	{
		return (*data)[s];
	}
	static value_type const & get_value(holder_type const & data, key_type s)
	{
		return (*data)[s];
	}
};

template<typename TV> struct container_traits<std::shared_ptr<TV>>
{
	typedef std::shared_ptr<TV> container_type;
	typedef TV value_type;
	typedef std::shared_ptr<TV> holder_type;

	static constexpr bool is_associative_container = false;

	static holder_type allocate(size_t s)
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

	static value_type & get_value(holder_type & data, size_t s)
	{
		return data.get()[s];
	}
	static value_type const & get_value(holder_type const & data, size_t s)
	{
		return data.get()[s];
	}
};
/** @}*/
}
// namespace simpla
#endif /* CONTAINER_TRAITS_H_ */
