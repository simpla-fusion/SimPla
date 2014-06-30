/*
 * conatiner_dense.h
 *
 *  Created on: 2014年6月30日
 *      Author: salmon
 */

#ifndef CONATINER_DENSE_H_
#define CONATINER_DENSE_H_

#include <mutex>

#include "../utilities/log.h"
#include "../utilities/memory_pool.h"

namespace simpla
{

template<typename TKey, typename TV>
class DenseContainer
{
	std::mutex write_lock_;
public:

	typedef TV value_type;
	typedef TKey key_type;
	typedef DenseContainer<key_type, value_type> this_type;

	static constexpr bool is_dense_storage = true;

	std::shared_ptr<value_type> data_;

	std::function<size_t(key_type)> hash_fun_;

	size_t num_of_ele_;

	const value_type default_value_;

	template<typename TR, typename ... Others>
	DenseContainer(TR const& range, std::function<size_t(key_type)> const & hash, value_type d, Others && ...) :
			data_(nullptr),

			hash_fun_(hash),

			num_of_ele_(std::distance(std::get<0>(range), std::get<1>(range))),

			default_value_(d)
	{
	}

	template<typename TR, typename ... Others>
	DenseContainer(TR const& range, std::function<size_t(key_type)> const & hash, Others && ...others) :
			DenseContainer(range, hash, value_type() * 0, std::forward<Others>(others)...)
	{
	}

	DenseContainer(this_type const & rhs) :
			data_(rhs.data_), hash_fun_(rhs.hash_fun_), num_of_ele_(rhs.num_of_ele_), default_value_(rhs.default_value_)
	{
	}

	DenseContainer(this_type &&rhs) :
			data_(std::forward<std::shared_ptr<value_type>>(rhs.data_)),

			hash_fun_(std::forward<std::function<size_t(key_type)> >(rhs.hash_fun_)),

			num_of_ele_(rhs.num_of_ele_), default_value_(rhs.default_value_)
	{
	}

	~DenseContainer()
	{
	}

	void swap(this_type & rhs)
	{
		data_.swap(rhs.data_);

		std::swap(hash_fun_, rhs.hash_fun_);
		std::swap(num_of_ele_, rhs.num_of_ele_);
		std::swap(default_value_, rhs.default_value_);

	}
	bool empty() const
	{
		return data_ == nullptr;
	}
	size_t size() const
	{
		return empty() ? 0 : num_of_ele_;
	}
	value_type const & default_value() const
	{
		return default_value_;
	}
	void allocate()
	{
		if (data_ == nullptr)
		{
			data_ = MEMPOOL.allocate_shared_ptr< value_type> (num_of_ele_);
		}
	}

	void initialize()
	{
		if(data_ == nullptr)
		{
			allocate();

			fill(default_value_);
		}
	}

	void clear()
	{
		data_=nullptr;
	}

	void fill(value_type v)
	{
		allocate();

		for(size_t i=0;i<num_of_ele_;++i)
		{
			data_.get()[i]=v;
		}
	}

	void lock()
	{
		write_lock_.lock();
	}
	void unlock()
	{
		write_lock_.unlock();
	}

	value_type & at(key_type s)
	{
		auto idx=hash_fun_(s);

		if (idx<0 ||idx>=num_of_ele_) OUT_RANGE_ERROR("["+ToString(0)+"~"+ToString(num_of_ele_)+"]");

		return data_.get()[idx];
	}
	value_type const & at(key_type s) const
	{
		auto idx=hash_fun_(s);

		if (idx<0 ||idx>=num_of_ele_) OUT_RANGE_ERROR("["+ToString(0)+"~"+ToString(num_of_ele_)+"]");

		return data_.get()[idx];

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
		return data_.get()[hash_fun_(s)];
	}

	inline value_type const & get(key_type s) const
	{
		return data_.get()[hash_fun_(s)];
	}

};

}
 // namespace simpla

#endif /* CONATINER_DENSE_H_ */
