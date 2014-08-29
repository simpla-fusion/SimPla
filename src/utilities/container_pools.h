/**
 * \file container_pools.h
 *
 * \date    2014年8月26日  下午4:30:23 
 * \author salmon
 */

#ifndef CONTAINER_POOLS_H_
#define CONTAINER_POOLS_H_

namespace simpla
{

template<typename IndexType, typename ValueType, typename InnerAlloc = std::allocator<ValueType> >
class ContainerPool: public std::map<IndexType, std::list<ValueType, InnerAlloc>>
{

	std::mutex write_lock_;

public:

	typedef IndexType key_type;
	typedef ValueType value_type;
	typedef Alloc allocator_type;

	typedef std::list<value_type, Alloc> sub_pool_type;

	typedef std::map<key_type, sub_pool_type> base_type;

	typedef ContainerPool<key_type, value_type, allocator_type> this_type;

	//container

private:

	sub_pool_type defaut_value_;

public:

	//***************************************************************************************************
	// Constructor

	ContainerPool()
	{
	}

	ContainerPool(allocator_type const & alloc) :
			defaut_value_(alloc)
	{
	}

	// Destructor
	~ContainerPool()
	{
		base_type::clear();
	}

	this_type clone() const
	{
		return std::move(ContainerPool(get_allocator()));
	}
	//***************************************************************************************************

	allocator_type get_allocator() const
	{
		return defaut_value_.get_allocator();
	}

	/**
	 *
	 * @return a new child container with shared allocator
	 */
	value_type create_child() const
	{
		return std::move(sub_pool_type(get_allocator()));
	}

	void lock()
	{
		write_lock_.lock();
	}
	void unlock()
	{
		write_lock_.unlock();
	}

	sub_pool_type const & default_value() const
	{
		return defaut_value_;
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

	sub_pool_type & at(key_type s)
	{
		return get(s);
	}
	sub_pool_type const & at(key_type s) const
	{
		return get(s);
	}
	sub_pool_type & operator[](key_type s)
	{
		return get(s);
	}
	sub_pool_type & operator[](key_type s) const
	{
		return get(s);
	}
	inline sub_pool_type & get(key_type s)
	{
		sub_pool_type res;
		auto it = base_type::find(s);
		if (it == base_type::end())
		{
			base_type::emplace(s, create_child());
		}

		return base_type::operator[](s);
	}

	inline sub_pool_type const & get(key_type s) const
	{
		auto it = base_type::find(s);
		if (it == base_type::end())
		{
			return defaut_value_;
		}
		else
		{
			return it->second;
		}
	}

	void merge(this_type & other);

	void add(sub_pool_type &src);

	template<typename TRange>
	void remove(TRange && index_range);

	template<typename TRange, typename TPredFun>
	void remove(TRange && index_range, TPredFun && pred_fun);

	template<typename TRange, typename TPredFun>
	void modify(TRange && index_range, TPredFun && pred_fun);

	template<typename TRange, typename TRes, typename Func, typename Reduction>
	void reduce(TRange && index_range, TRes * res, Func && fun, Reduction &&) const;

	template<typename TRange, typename HashFun>
	void sort(TRange && index_range, HashFun const &);

	size_t count() const;

	template<typename TRange>
	size_t count(TRange && index_range) const;

};

template<typename IndexType, typename ValueType, typename Alloc>
template<typename TR>
size_t ContainerPool<IndexType, ValueType, Alloc>::count(TR && range) const
{

	VERBOSE << "Count Particles";

	size_t count = 0;

	for (auto s : range)
	{
		auto it = base_type::find(s);

		if (it != base_type::end()) count += it->second.size();
	}

	return count;
}
template<typename IndexType, typename ValueType, typename Alloc>
size_t ContainerPool<IndexType, ValueType, Alloc>::count() const
{

	return count(mesh.Select(IForm));
}

template<typename IndexType, typename ValueType, typename Alloc>
template<typename TRange, typename HashFun>
void ContainerPool<IndexType, ValueType, Alloc>::sort(TRange && range, HashFun const & hash)
{

	parallel_reduce(range, *this, this,

	[this](TRange const & r,this_type * t_buffer)
	{
		auto it = this->base_type::find(s);

		if (it == this->base_type::end()) continue;

		auto pt = it->second.begin();

		while (pt != it->second.end())
		{
			auto p = pt;
			++pt;

			auto gid = hash(*p);
			if (gid != s)
			{
				auto & dest = t_buffer->get( gid );
				dest->splice(dest->begin(), it->second, p);
			}

		}

	},

	[&](this_type & l,this_type * r)
	{
		r->merge(l);
	});

}

template<typename IndexType, typename ValueType, typename Alloc>
void ContainerPool<IndexType, ValueType, Alloc>::clear_empty()
{
	auto it = base_type::begin(), ie = base_type::end();

	while (it != ie)
	{
		auto t = it;
		++it;
		if (t->second.empty())
		{
			base_type::erase(t);
		}
	}
}

template<typename IndexType, typename ValueType, typename Alloc>
void ContainerPool<IndexType, ValueType, Alloc>::merge(this_type & other)
{
	for (auto & v : other)
	{
		auto & c = this->get(v.first);
		c.splice(c.begin(), v.second);
	}

}
template<typename IndexType, typename ValueType, typename Alloc>
void ContainerPool<IndexType, ValueType, Alloc>::add(sub_pool_type &other)
{
	if (other->size() <= 0) return;

	parallel_reduce(other, *this, this,

	[](TRange && r)
	{

	},

	[](this_type & l ,this_type * r)
	{
		r->merge(l);
	}

	);
}

template<typename IndexType, typename ValueType, typename Alloc>
template<typename TRange>
void ContainerPool<IndexType, ValueType, Alloc>::remove(TRange && index_range)
{

	parallel_reduce(std::forward<TRange>(index_range), defaut_value_, &defaut_value_,

	[&](TRange && r,sub_pool_type * t_buffer)
	{
		for(auto & s:r)
		{
			auto cell_it = base_type::find(s);

			if (cell_it == base_type::end())
			continue;

			t_buffer->splice(t_buffer->begin(), cell_it->second);
		}
	},

	[](sub_pool_type & l, sub_pool_type* r)
	{
		l.clear();
	}

	);
}
template<typename IndexType, typename ValueType, typename Alloc>
template<typename TRange, typename TPredFun>
void ContainerPool<IndexType, ValueType, Alloc>::remove(TRange && index_range, TPredFun && pred_fun)
{

	parallel_reduce(std::forward<TRange>(index_range), defaut_value_, &defaut_value_,

	[&](TRange && r,sub_pool_type * t_buffer)
	{

		for(auto const & s:r)
		{
			auto cell_it = base_type::find(s);

			if (cell_it == base_type::end())
			continue;

			auto it = cell_it->second.begin();
			auto ie = cell_it->second.end();

			do
			{
				auto it_p = it;
				++it;

				if (pred_fun(*it_p))
				{
					t_buffer->splice(t_buffer->begin(), cell_it->second, it_p);
				}

			}while (it != ie);
		}

	},

	[](sub_pool_type & l, sub_pool_type* r)
	{
		l.clear();
	}

	);

}

template<typename IndexType, typename ValueType, typename Alloc> template<typename TRange, typename Func>
void ContainerPool<IndexType, ValueType, Alloc>::modify(TRange && index_range, Func && fun)
{

	parallel_for(std::forward<TRange>(index_range),

	[&](TRange && r)
	{
		for(auto & gid:r)
		{
			auto it = this->base_type::find(gid);
			if (it != this->base_type::end())
			{
				for (auto & p : it->second)
				{
					fun(&p);
				}
			}
		}
	}

	);

}
template<typename IndexType, typename ValueType, typename Alloc>
template<typename TRange, typename TRes, typename Func, typename Reduction>
void ContainerPool<IndexType, ValueType, Alloc>::reduce(TRange && index_range, TRes * res, Func && fun,
        Reduction && reduce) const
{
	parallel_reduce(std::forward<TRange>(index_range), defaut_value_, res,

	[&](TRange & r,TRes * buffer)
	{
		for(auto &gid:r)
		{
			auto it = this->base_type::find(gid);
			if (it != this->base_type::end())
			{
				for (auto const & p : it->second)
				{
					fun( p,buffer);
				}
			}
		}
	}

	, std::forward<Reduction>(reduce)

	);
}
}  // namespace simpla

#endif /* CONTAINER_POOLS_H_ */
