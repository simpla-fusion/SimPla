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

namespace simpla
{
template<typename ...>struct ContainerPool;

template<typename KeyType, typename ValueType>
class ContainerPool<KeyType, ValueType>
{
public:
	typedef ValueType value_type;
	typedef KeyType key_type;
	typedef ContainerPool<key_type, value_type> this_type;

	typedef std::function<key_type(value_type const &)> hash_func;
private:
	typedef std::list<value_type> inner_container;
	typedef typename inner_container::pointer inner_pointer;
	typedef std::map<key_type, inner_pointer> map_container;

	typedef map_container storage_type;

	hash_func hash_;

	inner_container trash_;

	map_container data_;

public:

	//***************************************************************************************************
	// Constructor

	ContainerPool()
	{
	}

	ContainerPool(this_type const & other) :
			trash_(other.trash_), hash_(other.hash_)

	{
	}

	// Destructor
	~ContainerPool()
	{
	}

	inner_container make_buffer() const
	{
		return std::move(inner_container(trash_));
	}

	size_t size() const;
	bool empty() const
	{
		return data_.empty();
	}

	void insert(value_type &&);
	void insert(inner_container &);
	void insert(this_type &);

	void sort();
	template<typename THash>
	void sort(THash const &);

	template<typename PredFun>
	void remove_if(PredFun);

	void remove(key_type const& s);

	template<typename Func>
	void modify(key_type const & s, Func const & func, inner_container*buffer =
			nullptr);

	void clear()
	{
		//TODO move every thing into the trash
		data_.clear();
	}
	inner_container & find(key_type const& s)
	{
		auto it = data_.find(s);
		if (it == data_.end())
		{
			return trash_;
		}
		else
		{
			return it->second;
		}

	}
};

template<typename KeyType, typename ValueType>
size_t ContainerPool<KeyType, ValueType>::size() const
{
	size_t count = 0;

	for (auto const &item : trash_)
	{
		count += item.second.size();
	}

	return count;
}

template<typename KeyType, typename ValueType>
void ContainerPool<KeyType, ValueType>::insert(value_type && p)
{
	trash_[hash_(p)].push_back(p);
}

template<typename KeyType, typename ValueType>
void ContainerPool<KeyType, ValueType>::insert(this_type & other)
{
	for (auto & v : other.trash_)
	{
		auto & c = trash_[v.first];
		c.splice(c.begin(), v.second);
	}

}
template<typename KeyType, typename ValueType>
void ContainerPool<KeyType, ValueType>::sort()
{

	auto range = make_range(trash_.begin(), trash_.end());

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
		r->insert(l);
	});

}

template<typename KeyType, typename ValueType>
template<typename Func>
void ContainerPool<KeyType, ValueType>::modify(key_type const & s,
		Func const & func, inner_container*buffer)
{

	auto t_buffer = make_buffer();

	auto cell = data_.find(s);
	if (cell != data_.end())
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

}  // namespace simpla

#endif /* CONTAINER_POOL_H_ */
