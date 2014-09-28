/**
 * \file container_pool.h
 *
 * \date    2014年8月26日  下午4:30:23 
 * \author salmon
 */

#ifndef CONTAINER_POOL_H_
#define CONTAINER_POOL_H_
#include <map>
#include <list>
#include "../utilities/range.h"

namespace simpla
{

template<typename ValueType, typename HashFunc>
class ContainerPool
{
public:
	typedef ValueType value_type;
	typedef HashFunc hash_func;
	typedef decltype(std::declval<hash_func>()(std::declval<value_type>())) key_type;
private:
	typedef std::vector<value_type> inner_container;
	typedef typename inner_container::pointer inner_pointer;
	typedef std::unordered_map<key_type, inner_pointer> map_container;

	inner_container data_;

	map_container hash_map;

	inner_container_type default_value_;
	storage_type_ data_;
	hash_func hash_;
public:

	//***************************************************************************************************
	// Constructor

	ContainerPool(hash_func hash)
			: hash_(hash)

	{
	}

	ContainerPool(this_type const & other)
			: data_(other.data_), hash_(other.hash_)

	{
	}

	// Destructor
	~ContainerPool()
	{
	}

	inner_container_type make_buffer() const
	{
		return std::move(inner_container_type(default_value_));
	}

	void insert(inner_container_type &);

	size_t size() const;

	void insert(value_type &&);
	void sort();
	void merge(this_type & other);

	template<typename PredFun>
	void remove_if(PredFun);

	void remove(key_type const& s);

	template<typename Func>
	void modify(key_type const & s, Func const & func, inner_container_type*buffer = nullptr);

	bool empty() const
	{
		return data_.empty();
	}

	void clear()
	{
		data_.clear();
	}
	inner_container_type & find(key_type const& s)
	{
		auto it = data_.find(s);
		if (it == data_.end())
		{
			return default_value_;
		}
		else
		{
			return it->second;
		}

	}
};

template<typename ValueType, typename HashFunc>
size_t ContainerPool<ValueType, HashFunc>::size() const
{
	size_t count = 0;

	for (auto const &item : data_)
	{
		count += item.second.size();
	}

	return count;
}

template<typename ValueType, typename HashFunc>
void ContainerPool<ValueType, HashFunc>::insert(value_type && p)
{
	data_[hash_(p)].push_back(p);
}

template<typename ValueType, typename HashFunc>
void ContainerPool<ValueType, HashFunc>::merge(this_type & other)
{
	for (auto & v : other.data_)
	{
		auto & c = data_[v.first];
		c.splice(c.begin(), v.second);
	}

}
template<typename ValueType, typename HashFunc>
void ContainerPool<ValueType, HashFunc>::sort()
{

	auto range = make_range(data_.begin(), data_.end());

	parallel_reduce((range), *this, this,

	[&](decltype(range) const & r,this_type * t_buffer)
	{
		for(auto const & item:r)
		{

			auto pt=item.second.begin();
			while(pt!=item.second.end())
			{
				auto p = pt;
				++pt;

				auto gid = hash_(*p);

				if (gid != item.first)
				{
					auto & dest = (*t_buffer)[gid];
					dest->splice(dest->begin(), item.second, p);
				}
			}

		}
	},

	[&](this_type & l,this_type * r)
	{
		r->merge(l);
	});

}

template<typename ValueType, typename HashFunc>
template<typename TRange, typename Func>
void ContainerPool<ValueType, HashFunc>::modify(key_type const & s, Func const & func, inner_container_type*buffer)
{

	auto t_buffer = make_buffer();

	auto cell = storage_type_::find(s)
	if (cell != storage_type_::end())
	{
		auto pt = cell->second->begin();
		while (pt != cell->second->end())
		{
			auto p = pt;
			++pt;

			if (fun(&(*p)))
			{
				t_buffer->splice(t_buffer->begin(), cell->second, p);
			}
		}
	}

	if (buffer != nullptr)
		buffer->splice(buffer->begin(), t_buffer);

}
template<typename ValueType, typename HashFunc>
template<typename TRange>
void ContainerPool<ValueType, HashFunc>::remove(TRange && index_range)
{
	modify(std::forward<TRange>(index_range), [](value_type *)
	{	return true;});
}

}  // namespace simpla

#endif /* CONTAINER_POOL_H_ */
