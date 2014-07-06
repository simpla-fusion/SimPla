/*
 * container_sparse.h
 *
 *  Created on: 2014-6-30
 *      Author: salmon
 */

#ifndef CONTAINER_SPARSE_H_
#define CONTAINER_SPARSE_H_

#include <mutex>

#include "../utilities/log.h"
#include "../utilities/sp_complex.h"

namespace simpla
{
/**
 * \ingroup DataStruct
 * \brief  Sparse Container
 *
 */
template<typename TI, typename TV>
class SparseContainer: public std::map<TI, TV>
{
	std::mutex write_lock_;

public:

	typedef TV value_type;
	typedef TI key_type;

	typedef SparseContainer<key_type, value_type> this_type;

	static constexpr bool is_dense_storage = false;

	typedef std::map<key_type, value_type> base_container_type;

	const value_type default_value_;

	SparseContainer(value_type d = value_type(0))
			: base_container_type(), default_value_(d)
	{

	}
	template<typename TR, typename THash, typename ...Others>
	SparseContainer(TR const &, THash const&, Others && ... others)
			: base_container_type(std::forward<Others>(others)...), default_value_(value_type())
	{
	}
	template<typename TR, typename THash, typename ...Others>
	SparseContainer(TR const &, THash const&, value_type d, Others && ... others)
			: base_container_type(std::forward<Others>(others)...), default_value_(d)
	{
	}

	SparseContainer(this_type const & rhs)
			: base_container_type(rhs), default_value_(rhs.default_value_)
	{
	}

	SparseContainer(this_type &&rhs)
			: base_container_type(std::forward<base_container_type>(rhs)), default_value_(rhs.default_value_)
	{
	}

	~SparseContainer()
	{
	}

	this_type & operator =(this_type const & rhs)
	{
		base_container_type::operator=(rhs);

		return (*this);
	}
	void swap(this_type & rhs)
	{
		base_container_type::swap(rhs);

		std::swap(default_value_, rhs.default_value_);
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

	size_t size() const
	{
		return base_container_type::size();
	}

	void allocate()
	{
	}

	void initialize()
	{
	}

	void clear()
	{
		base_container_type::clear();
	}

	void fill(value_type v)
	{
		for (auto & item : *this)
		{
			item.second = v;
		}
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
			base_container_type::operator[](s) = default_value_;
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

#endif /* CONTAINER_SPARSE_H_ */
