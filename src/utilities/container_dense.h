/*
 * conatiner_dense.h
 *
 *  created on: 2014-6-30
 *      Author: salmon
 */

#ifndef CONATINER_DENSE_H_
#define CONATINER_DENSE_H_

#include <mutex>
#include <cstring>
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
#include "../utilities/utilities.h"
namespace simpla
{
/**
 * \ingroup DataStruct
 * \brief  Dense Container
 *
 */
template<typename TKey, typename TV>
class DenseContainer
{
	std::mutex write_lock_;
public:

	typedef TV value_type;
	typedef TKey key_type;
	typedef DenseContainer<key_type, value_type> this_type;

	typedef value_type & reference;
	typedef value_type * pointer;

	static constexpr bool is_dense_storage = true;

	std::shared_ptr<value_type> data_;

	std::function<size_t(key_type)> hash_fun_;

	size_t num_of_ele_;

	const value_type default_value_;

	DenseContainer(value_type d = value_type())
			: num_of_ele_(0), default_value_(d)
	{
		hash_fun_ = [](key_type)
		{
			UNIMPLEMENT;

			return 0;
		};

	}
	template<typename TR, typename ... Others>
	DenseContainer(TR const& range, size_t max_hash_value, std::function<size_t(key_type)> const & hash, value_type d,
	        Others && ...)
			: data_(nullptr),

			hash_fun_(hash),

			num_of_ele_(max_hash_value),

			default_value_(d)
	{
	}

	template<typename TR, typename ... Others>
	DenseContainer(TR const& range, size_t max_hash_value, std::function<size_t(key_type)> const & hash,
	        Others && ...others)
			: DenseContainer(range, max_hash_value, hash, value_type(), std::forward<Others>(others)...)
	{
	}

	DenseContainer(this_type const & rhs)
			: data_(rhs.data_), hash_fun_(rhs.hash_fun_), num_of_ele_(rhs.num_of_ele_), default_value_(
			        rhs.default_value_)
	{
	}

	DenseContainer(this_type &&rhs)
			: data_(std::forward<std::shared_ptr<value_type>>(rhs.data_)),

			hash_fun_(std::forward<std::function<size_t(key_type)> >(rhs.hash_fun_)),

			num_of_ele_(rhs.num_of_ele_), default_value_(rhs.default_value_)
	{
	}

	virtual ~DenseContainer()
	{
	}
	const std::shared_ptr<value_type> data() const
	{
		return data_;
	}
	std::shared_ptr<value_type> data()
	{
		return data_;
	}
	void swap(this_type & rhs)
	{
		std::swap(rhs.data_, data_);
		std::swap(rhs.hash_fun_, hash_fun_);
		std::swap(rhs.num_of_ele_, num_of_ele_);
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
		clear();
	}

	void clear()
	{
		allocate();

		std::memset(data_.get(), 0, sizeof(value_type)*num_of_ele_);
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
		return data_.get()[hash_fun_(s)%num_of_ele_];
	}

	inline value_type const & get(key_type s) const
	{
		return data_.get()[hash_fun_(s)%num_of_ele_];
	}

};

}
 // namespace simpla

#endif /* CONATINER_DENSE_H_ */
