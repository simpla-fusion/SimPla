/*
 * container_container.h
 *
 *  Created on: 2014年7月1日
 *      Author: salmon
 */

#ifndef CONTAINER_CONTAINER_H_
#define CONTAINER_CONTAINER_H_

#include <map>
#include <list>
#include <mutex>

#ifndef NO_STD_CXX
//need  libstdc++
#include <ext/mt_allocator.h>
template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;
#endif

namespace simpla
{

template<typename TV> using ListOfSmallObject=std::list<TV/*,FixedSmallSizeAlloc<TV>*/>;


/**
 * @ingroup DataStruct
 * @brief  Container of container. i.e. std::map<TKey,std::list<TV>>, *  sub-containers share same allocator
 *
 *  index behavior:  operator[], at
 *   if key exist return reference of value at key;
 *   else return reference of default value;
 *
 */
template<typename TKey, typename TV, template<typename > class SubContainer = ListOfSmallObject>
class ContainerContainer: public std::map<TKey, SubContainer<TV>>
{
	std::mutex write_lock_;

public:

	typedef TKey key_type;

	typedef SubContainer<TV> value_type;

	typedef std::map<key_type, value_type> parent_container_type;

	typedef std::map<key_type, value_type> base_container_type;

	typedef ContainerContainer<key_type, value_type, SubContainer> this_type;

	typedef typename value_type::allocator_type allocator_type;

	//container

private:

	value_type default_value_;

public:

	//***************************************************************************************************
	// Constructor

	ContainerContainer()
	{
		CHECK("Construct");
	}

	ContainerContainer(allocator_type const & alloc)
			: default_value_(alloc)
	{
		CHECK("Construct");
	}

	// Destructor
	~ContainerContainer()
	{
		;
	}

	this_type clone() const
	{
		return std::move(ContainerContainer(get_allocator()));
	}
	//***************************************************************************************************

	allocator_type get_allocator() const
	{
		return default_value_.get_allocator();
	}

	/**
	 *
	 * @return a new child container with shared allocator
	 */
	value_type create_child() const
	{
		return std::move(value_type(get_allocator()));
	}

	void lock()
	{
		write_lock_.lock();
	}
	void unlock()
	{
		write_lock_.unlock();
	}

	value_type const & default_value() const
	{
		return default_value_;
	}

	//***************************************************************************************************

	size_t size() const
	{
		size_t res = 0;

		for (auto const & v : *this)
		{
			res += v.second.size();
		}
		return res;
	}

	value_type & at(key_type s)
	{
		return get(s);
	}
	value_type const & at(key_type s) const
	{
		return get(s);
	}
	value_type & operator[](key_type s)
	{
		return get(s);
	}
	value_type & operator[](key_type s) const
	{
		return get(s);
	}
	inline value_type & get(key_type s)
	{
		value_type res;
		auto it = base_container_type::find(s);
		if (it == base_container_type::end())
		{
			base_container_type::emplace(s, create_child());
		}

		return base_container_type::operator[](s);
	}

	inline value_type const & get(key_type s) const
	{
		auto it = base_container_type::find(s);
		if (it == base_container_type::end())
		{
			return default_value_;
		}
		else
		{
			return it->second;
		}
	}

};

}  // namespace simpla

#endif /* CONTAINER_CONTAINER_H_ */
