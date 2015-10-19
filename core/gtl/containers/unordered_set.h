/*
 * @file unordered_set.h
 *
 *  created on: 2014-4-29
 *      Author: salmon
 */

#ifndef CORE_GTL_CONTAINER_SORTED_SET_H_
#define CORE_GTL_CONTAINER_SORTED_SET_H_

#include <stddef.h>
#include <forward_list>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <set>
#include "../../dataset/dataset.h"
#include "../iterator/sp_indirect_iterator.h"
#include "../../parallel/distributed_object.h"

namespace simpla
{
/**
 * @brief an alternative implement of `std::unordered_multiset`
 * - elements are organized into buckets,
 * - optimized for frequently rehash,insert and remove
 * -
 *
 *>  Unordered multi-set is an associative container that contains set
 *> of possibly non-unique objects of type Key. --http://en.cppreference.com/w/cpp/container/unordered_multiset
 */
template<typename T, typename BucketKeyType=size_t, typename Allocator = std::allocator<T> >
class UnorderedSet
{

public:
	typedef T value_type;
	typedef BucketKeyType key_type;
	typedef Allocator allocator_type;

	typedef UnorderedSet<value_type, key_type, allocator_type> this_type;

	typedef std::forward_list<T, allocator_type> bucket_type;

	typedef std::map<key_type, bucket_type> base_container_type;

//private:


	base_container_type m_data_;

	std::set<key_type> m_modified_;

public:

	typedef std::set<key_type> range_type;

	// Constructor
	UnorderedSet()
	{
	}

	UnorderedSet(this_type const &other) :
			m_data_(other.m_data_)
	{
	}

	UnorderedSet(this_type &&other) :
			m_data_(other.m_data_)
	{
	}

	template<typename TRange>
	UnorderedSet(this_type &other, TRange const &range)
	{
		splice(range, other);
	}

	~UnorderedSet()
	{
	}


	void swap(this_type &other)
	{
		m_data_.swap(other.m_data_);
	}

	template<typename OS> OS &print(OS &os) const
	{
		return os;
	}

	virtual DataSet dataset() const;

	virtual void dataset(DataSet const &);

	bool empty() const
	{
		return m_data_.empty();
	}


	bool is_divisible() const
	{
		return false;
	}

	bucket_type &operator[](key_type const &key)
	{
		m_modified_.insert(key);
		return m_data_[key];
	}

	bucket_type &at(key_type const &key)
	{
		m_modified_.insert(key);
		return m_data_.at(key);
	}

	bucket_type const &at(key_type const &key) const
	{
		return m_data_.at(key);
	}

	typedef typename bucket_type::iterator local_iterator;

	typedef typename bucket_type::const_iterator const_local_iterator;

	local_iterator begin(key_type const &id)
	{
		return std::move((*this)[id].begin());
	}

	local_iterator end(key_type const &id)
	{
		return std::move((*this)[id].end());
	}

	constexpr const_local_iterator cbegin(key_type const &id) const
	{
		return std::move(m_data_.at(id).begin());
	}

	constexpr const_local_iterator cend(key_type const &id) const
	{
		return std::move(m_data_.at(id).end());
	}

	constexpr const_local_iterator begin(key_type const &id) const
	{
		return std::move(m_data_.at(id).begin());
	}

	constexpr const_local_iterator end(key_type const &id) const
	{
		return std::move(m_data_.at(id).end());
	}

	template<typename TV>
	void insert(key_type const &key, TV &&v)
	{
		(*this)[key].insert_after(m_data_[key].before_begin(),
				std::forward<TV>(v));
	}

	template<typename TV, typename Hash>
	key_type insert(TV const &v, Hash const &hasher)
	{
		auto s = hasher(v);

		(*this)[s].push_front(v);

		return s;
	}

	template<typename TV, typename Hash>
	void push_front(TV &&v, Hash const &hasher)
	{
		(*this)[hasher(v)].push_front(std::forward<TV>(v));
	}

	void insert(std::initializer_list<value_type> ilist)
	{
		insert(ilist.begin(), ilist.end());
	}

	template<typename InputIter, typename Hash>
	void insert(InputIter first, InputIter last, Hash const &hasher)
	{
		for (auto it = first; it != last; ++it)
		{
			insert(*it, hasher);
		}
	}

	template<typename Hash>
	void insert(key_type const &key, std::initializer_list<value_type> ilist, Hash const &hasher)
	{
		(*this)[key].insert_after(m_data_[key].before_begin(), ilist);
	}

	template<typename InputIter, typename Hash>
	void insert(key_type const &key, InputIter first, InputIter last, Hash const &hasher)
	{
		(*this)[key].insert_after(m_data_[key].before_begin(), first, last);
	}

	template<typename ...Others>
	void assign(key_type const &key, Others &&...others)
	{
		(*this)[key].assign(std::forward<Others>(others)...);
	}

	template<typename IputIterator>
	void assign(IputIterator first, IputIterator last)
	{
		clear();
		insert(first, last);
	}

	template<typename TRange>
	void splice(TRange const &range, this_type &other)
	{
		for (auto const &key : range)
		{
			auto it = other.m_data_.find(key);
			if (it != other.m_data_.end())
			{
				auto &dest = m_data_[it->first];

				dest.splice_after(dest.before_begin(), it->seond);
			}
		}
	}

	/**
	 *  BucketInputIter=  std::pair<key_type,bucket_type> *
	 * @param first
	 * @param last
	 */
	template<typename BucketInputIter>
	void splice(BucketInputIter first, BucketInputIter last)
	{
		for (auto it = first; it != last; ++it)
		{
			auto &dest = m_data_[it->first];

			dest.splice_after(dest.before_begin(), it->second);
		}
	}

	void erase(key_type const &key)
	{
		m_data_.erase(key);
		m_modified_.erase(key);

	}

	void erase_all()
	{
		m_data_.clear();
		m_modified_.clear();
	}

	void clear()
	{
		erase_all();

	}

	template<typename TRange>
	void erase(TRange const &range)
	{
		for (auto const &key : range)
		{
			erase((key));
		}

	}
	/**
	 *  move  elements  which `hash(value)!=key`  from   `m_data_[key]`
	 *   to container `other[hash(value)]`
	 * @param key
	 * @param other
	 * @return number of moved elements
	 */
private:
	template<typename Hash>
	size_t rehash_one(key_type key, base_container_type &other, Hash const &hasher)
	{
		if (m_modified_.find(key) == m_modified_.end())
		{
			return 0;
		}

		m_modified_.erase(key);

		auto it = m_data_.find(key);
		if (it == m_data_.end())
		{
			return 0;
		}
		auto &bucket = it->second;
		auto pt = bucket.begin();
		size_t count = 0;
		while (pt != bucket.end())
		{
			auto p = pt;
			++pt;

			auto o_key = hasher(*p);

			if (o_key != key)
			{
				auto &dest = other[o_key];
				dest.splice_after(dest.before_begin(), bucket, p);
				++count;
			}
		}

		return count;
	}

public:
	template<typename TRange, typename Hash>
	void rehash(TRange const &r, Hash const &hasher)
	{
		base_container_type other;

		for (auto const &s : r)
		{
			rehash_one(s, other, hasher);
		}

		splice(other.begin(), other.end());

	}

	template<typename Hash>
	void rehash(Hash const &hasher)
	{
		rehash(m_modified_, hasher);
	}

	template<typename TFun>
	void for_each(TFun const &fun)
	{

		for (auto const &bucket : *this)
		{
			for (auto &p : bucket.second)
			{
				fun(p);
			}
		}

	}

	template<typename TFun>
	void erase_if(TFun const &fun, key_type hint = 0)
	{
		if (hint == 0)
		{
			for (auto const &item : *this)
			{
				erase_if(fun, item.first);
			}
		}
		else
		{
			auto &bucket = (*this)[hint];

			bucket.erase(std::remove_if(bucket.begin(), bucket.end(), fun),
					bucket.end());
		}

	}

	template<typename TFun>
	void modify(TFun const &fun, key_type hint = 0)
	{
		if (hint == 0)
		{
			for (auto const &item : *this)
			{
				modify(fun, item.first);
			}
		}
		else
		{
			if (this->find(hint) != this->end())
			{
				auto &bucket = (*this)[hint];

				for (auto &p : bucket)
				{
					fun(&p);
				}
			}
		}

	}

	long size(key_type const &key) const
	{
		long count = 0;

		auto item = m_data_.find(key);
		if (item != m_data_.end())
		{
			count = std::distance(item->second.begin(), item->second.end());
		}
		return count;
	}


	template<typename TRange>
	size_t size_all(TRange const &range) const
	{
		size_t count = 0;
		for (auto const &key : range)
		{
			count += size(key);
		}
		return count;
	}

	size_t size() const
	{

		size_t count = 0;
		for (auto const &item : m_data_)
		{
			count += std::distance(item.second.begin(), item.second.end());
		}
		return count;
	}

	typedef typename base_container_type::iterator iterator;
	typedef typename base_container_type::const_iterator const_iterator;

	const_iterator begin() const
	{
		return m_data_.cbegin();
	}

	const_iterator end() const
	{
		return m_data_.cend();
	}

	const_iterator cbegin() const
	{
		return m_data_.cbegin();
	}

	const_iterator cend() const
	{
		return m_data_.cend();
	}

	iterator find(key_type const &key)
	{
		auto it = this->m_data_.find(key);
		if (it != m_data_.end())
		{
			m_modified_.insert(key);
		}
		return std::move(it);
	}

	auto find(key_type const &key) const
	DECL_RET_TYPE(this->m_data_.find(key))

};


template<typename T, typename BucketKeyType, typename Allocator>
DataSet UnorderedSet<T, BucketKeyType, Allocator>::dataset() const
{
	DataSet res;

	res.datatype = traits::datatype<value_type>::create();
//	CHECK(res.datatype.name());
//	CHECK(res.datatype.is_valid());

	DataSpace::index_type count = size();

	res.data = sp_alloc_memory(count * sizeof(value_type));

	value_type *p = reinterpret_cast<value_type *>(res.data.get());

	NEED_OPTIMIZATION;

	for (auto const &item:m_data_)
	{
		for (auto const &pit : item.second)
		{
			*p = pit;
			++p;
		}

	}

	DataSpace(1, &count).swap(res.dataspace);


	return std::move(res);
}

template<typename T, typename BucketKeyType, typename Allocator>
void UnorderedSet<T, BucketKeyType, Allocator>::dataset(DataSet const &ds)
{

	value_type const *d = reinterpret_cast<value_type const *>(ds.data.get());

	NEED_OPTIMIZATION;

	auto &bucket = m_data_[0];

	for (size_t i = 0, ie = ds.dataspace.size(); i < ie; ++i)
	{
		bucket.push_front(d[i]);
	}
};
}
// namespace simpla

#endif /* CORE_GTL_CONTAINER_SORTED_SET_H_ */
