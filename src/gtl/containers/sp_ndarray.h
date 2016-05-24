/**
 * @file sp_ndarray.h
 *
 * @date 2015-3-20
 * @author salmon
 */

#ifndef CORE_GTL_CONTAINERS_SP_NDARRAY_H_
#define CORE_GTL_CONTAINERS_SP_NDARRAY_H_

#include <stddef.h>
#include <memory>

#include "../../dataset/dataset.h"
#include "../nTuple.h"

namespace simpla
{

template<typename T, size_t NDIMS>
class ndArray
{

public:
	typedef T value_type;

	static constexpr size_t ndims = NDIMS;

	typedef ndArray<value_type, ndims> this_type;

	typedef long index_type;

	typedef nTuple<index_type, NDIMS> index_tuple;
private:
	std::shared_ptr<value_type> m_data_;
	value_type * m_start_;

	index_tuple m_offset_

	nTuple<size_t, ndims> m_dimensions_;
	nTuple<size_t, ndims> m_strides_
public:

	ndArray()
			: m_start_(nullptr)
	{
	}
	ndArray(index_tuple const & d)
	{
		dimensions(d);
		deploy();
	}
	ndArray(this_type const & other)
			: m_data_(other.m_data_), m_start_(other.m_start_), m_dimensions_(
					other.m_dimensions_), m_offset_(other.m_offset_)
	{

	}
	ndArray(this_type && other)
			: m_data_(other.m_data_), m_start_(other.m_start_), m_dimensions_(
					other.m_dimensions_), m_offset_(other.m_offset_)
	{
	}
	~ndArray()
	{
	}

	void swap(this_type & other)
	{
		std::swap(m_data_, other.m_data_);
		std::swap(m_start_, other.m_start_);
		std::swap(m_offset_, other.m_offset_);
		std::swap(m_dimensions_, other.m_dimensions_);
		std::swap(m_strides_, other.m_strides_);
	}
	this_type & operator=(this_type const & other)
	{
		this_type(other).swap(*this);
		return *this;
	}
	template<typename TD>
	void dimensions(TD const & d)
	{
		m_dimensions_ = d;

		m_strides_[ndims - 1] = 1;

		if (ndims > 1)
		{
			for (int i = ndims - 2; i >= 0; --i)
			{
				m_strides_[i] = m_dimensions_[i + 1] * m_strides_[i + 1];
			}
		}
	}
	index_tuple const & dimensions() const
	{
		return m_dimensions_;
	}

	template<typename TD>
	void offset(TD const & d)
	{
		m_offset_ = d;
	}

	constexpr size_t size() const
	{
		return m_dimensions_[0] * m_strides_[0];
	}
	void fill(value_type const & v)
	{
		std::fill(m_start_, m_start_ + size(), v);
	}

	bool empty() const
	{
		return size() == 0 || m_data_ == nullptr;
	}
	index_tuple const & offset() const
	{
		return m_offset_;
	}

	void deploy()
	{
		if (m_data_ == nullptr)
		{
			m_data_ = sp_make_shared_array<value_type>(size());
		}
		m_start_ = m_data_.get()
	}

	ndArray<value_type, ndims - 1> operator[](index_type s)
	{
		ndArray<value_type, ndims - 1> res;
		res.m_data_ = m_data_;
		res.m_dimensions_ = &m_dimensions_[1];
		res.m_offset_ = &m_offset_[1];
		res.m_strides_ = &m_strides_[1];
		res.m_start_ = m_start_ + s * m_strides_[0];
		return std::move(res);
	}
	value_type & operator[](index_tuple s)
	{
		return m_start_[inner_product(s - m_offset_, m_strides_)];
	}
	value_type & operator[](index_tuple s) const
	{
		return m_start_[inner_product(s - m_offset_, m_strides_)];
	}
	value_type & operator()(index_type ... s)
	{
		return m_start_[hash(s...)];
	}
	value_type const& operator()(index_type ... s) const
	{
		return m_start_[hash(s...)];
	}

	DataSet dataset() const
	{
		return make_dataset(m_data_, ndims, &m_dimensions_[0]);
	}
private:
	constexpr size_t hash(index_type s, index_type ...other) const
	{
		return ((s + m_dimensions_[ndims - 1 - sizeof...(other)]
				- m_offset_[ndims - 1 - sizeof...(other)])
				% m_dimensions_[ndims - 1 - sizeof...(other)])
				* m_strides_[ndims - 1 - sizeof...(other)] + hash(other...);
	}

	constexpr size_t hash(index_type s) const
	{
		return ((s + m_dimensions_[ndims - 1] - m_offset_[ndims - 1])
				% m_dimensions_[ndims - 1]) * m_strides_[ndims - 1];
	}
};

template<typename T>
class ndArray<T, 1>
{

public:
	typedef T value_type;
	static constexpr size_t ndims = 1;
	typedef long index_type;
	typedef long index_tuple;
private:
	std::shared_ptr<value_type> m_data_;
	value_type * m_start_;
	index_type m_dimensions_;
	index_type m_offset_;

public:
	ndArray()
			: m_data_(nullptr), m_start_(nullptr), m_dimensions_(0), m_offset_(
					0)
	{
	}
	void dimensions(index_tuple const & d)
	{
		m_dimensions_ = d;
	}
	index_tuple const & dimensions() const
	{
		return m_dimensions_;
	}
	void offset(index_tuple const & d)
	{
		m_offset_ = d;
	}
	index_tuple const & offset() const
	{
		return m_offset_;
	}

	void deploy()
	{
		if (m_data_ == nullptr)
		{
			m_data_ = sp_make_shared_array<value_type>(m_dimensions_);
		}
		m_start_ = m_data_.get()
	}

	value_type & operator[](index_type s)
	{
		return std::move(m_start_[s - m_offset_]);
	}
	value_type const & operator[](index_type s) const
	{
		return std::move(m_start_[s - m_offset_]);
	}

	value_type & operator()(index_type s)
	{
		return std::move(m_start_[s - m_offset_]);
	}
	value_type const & operator()(index_type s) const
	{
		return std::move(m_start_[s - m_offset_]);
	}

	DataSet dataset() const
	{
		return make_dataset(m_start_, ndims, &m_dimensions_);
	}
};

template<typename T, size_t N> size_t ndArray<T, N>::ndims;
}  // namespace simpla

#endif /* CORE_GTL_CONTAINERS_SP_NDARRAY_H_ */
