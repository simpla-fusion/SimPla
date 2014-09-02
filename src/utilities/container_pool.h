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
#include <scoped_allocator>

#ifdef NO_STD_CXX
template<typename T> using FixedSmallSizeAlloc=std::allocator<T>;
#else
//need  libstdc++
#include <ext/mt_allocator.h>
template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;
#endif

namespace simpla
{

template<typename KeyType, typename ValueType>
class ContainerPool: public std::map<KeyType, std::list<ValueType, __gnu_cxx ::__mt_alloc<ValueType> >,
        std::less<KeyType>,
        std::scoped_allocator_adaptor<
                std::allocator<std::pair<const KeyType, std::list<ValueType, __gnu_cxx ::__mt_alloc<ValueType> > > >,
                __gnu_cxx ::__mt_alloc<ValueType> > >
{

public:

	typedef KeyType key_type;
	typedef ValueType value_type;

private:
	typedef __gnu_cxx ::__mt_alloc<value_type> inner_allocator_type;

	typedef std::list<value_type, inner_allocator_type> inner_container_type;

	typedef std::map<key_type, inner_container_type, std::less<key_type>,
	        std::scoped_allocator_adaptor<std::allocator<std::pair<const key_type, inner_container_type> >,
	                inner_allocator_type> > base_type;

	typedef ContainerPool<key_type, value_type> this_type;

	inner_container_type default_value_;

public:

	//***************************************************************************************************
	// Constructor

	ContainerPool()
	{
	}

	ContainerPool(this_type const & other)
			: base_type(other)
	{
	}

	// Destructor
	~ContainerPool()
	{
		base_type::clear();
	}

	/**
	 *
	 * @return a new child container with shared allocator
	 */
	inner_container_type create_child() const
	{
		return std::move(inner_container_type(default_value_));
	}

	//***************************************************************************************************

	void merge(this_type & other);

	void add(inner_container_type &src);

	template<typename TRange>
	void remove(TRange && index_range);

	template<typename TRange, typename TPredFun>
	void remove(TRange && index_range, TPredFun && pred_fun);

	template<typename TRange, typename TPredFun>
	void modify(TRange && index_range, TPredFun && pred_fun);

	template<typename TRange, typename TRes, typename Func, typename Reduction>
	void reduce(TRange && index_range, TRes const & identity, TRes * res, Func && fun, Reduction &&) const;

	template<typename TRange, typename TRes, typename Func>
	void reduce(TRange && index_range, TRes const & identity, TRes * res, Func && fun) const;

	template<typename TRange, typename HashFun>
	void sort(TRange && index_range, HashFun const &);

	size_t size() const
	{
		return count();
	}

	size_t count() const;

	template<typename TRange>
	size_t count(TRange && index_range) const;

};

template<typename IndexType, typename ValueType>
template<typename TR>
size_t ContainerPool<IndexType, ValueType>::count(TR && range) const
{

	size_t count = 0;

	for (auto s : range)
	{
		auto it = base_type::find(s);

		if (it != base_type::end())
			count += it->second.size();
	}

	return count;
}

template<typename IndexType, typename ValueType>
template<typename TRange, typename HashFun>
void ContainerPool<IndexType, ValueType>::sort(TRange && range, HashFun const & hash)
{

	parallel_reduce(range, *this, this,

	[&](TRange const & r,this_type * t_buffer)
	{
		for(auto const & s:r)
		{
			auto it = this->base_type::find(s);

			if (it != this->base_type::end())
			{

				auto pt = it->second.begin();

				while (pt != it->second.end())
				{
					auto p = pt;
					++pt;

					auto gid = hash(*p);
					if (gid != s)
					{
						auto & dest = t_buffer[gid];
						dest->splice(dest->begin(), it->second, p);
					}

				}
			}
		}

	},

	[&](this_type & l,this_type * r)
	{
		r->merge(l);
	});

}

//template<typename IndexType, typename ValueType>
//void ContainerPool<IndexType, ValueType>::clear_empty()
//{
//	auto it = base_type::begin(), ie = base_type::end();
//
//	while (it != ie)
//	{
//		auto t = it;
//		++it;
//		if (t->second.empty())
//		{
//			base_type::erase(t);
//		}
//	}
//}

template<typename IndexType, typename ValueType>
void ContainerPool<IndexType, ValueType>::merge(this_type & other)
{
	for (auto & v : other.data_)
	{
		auto & c = base_type::operator[](v.first);
		c.splice(c.begin(), v.second);
	}

}
template<typename IndexType, typename ValueType>
void ContainerPool<IndexType, ValueType>::add(inner_container_type &other)
{
	if (other->size() <= 0)
		return;

}

template<typename IndexType, typename ValueType>
template<typename TRange>
void ContainerPool<IndexType, ValueType>::remove(TRange && index_range)
{

	parallel_reduce(std::forward<TRange>(index_range), default_value_, &default_value_,

	[&](TRange && r,inner_container_type * t_buffer)
	{
		for(auto & s:r)
		{
			auto cell_it = base_type::find(s);

			if (cell_it == base_type::end())
			continue;

			t_buffer->splice(t_buffer->begin(), cell_it->second);
		}
	},

	[](inner_container_type & l, inner_container_type* r)
	{
		l.clear();
	}

	);
}

template<typename IndexType, typename ValueType> template<typename TRange, typename Func>
void ContainerPool<IndexType, ValueType>::modify(TRange && index_range, Func && fun)
{

	parallel_for(std::forward<TRange>(index_range),

	[&](TRange && r)
	{
		for(auto & gid:r)
		{
			auto it = this->base_type::find(gid);
			if (it != this->base_type::end())
			{
				fun(it);
			}
		}
	}

	);

}
template<typename IndexType, typename ValueType>
template<typename TRange, typename TRes, typename Func, typename Reduction>
void ContainerPool<IndexType, ValueType>::reduce(TRange && index_range, TRes const & identity, TRes * res, Func && fun,
        Reduction && reduce) const
{
	parallel_reduce(std::forward<TRange>(index_range), identity, res,

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

template<typename IndexType, typename ValueType>
template<typename TRange, typename TRes, typename Func>
void ContainerPool<IndexType, ValueType>::reduce(TRange && index_range, TRes const & identity, TRes * res,
        Func && fun) const
{
	parallel_reduce(std::forward<TRange>(index_range), identity, res, std::forward<Func>(fun), [](TRes & l,TRes *r)
	{	*r+=l;});
}
}  // namespace simpla

#endif /* CONTAINER_POOL_H_ */
