/**
 * @file sp_ndarray_iterator.h
 *
 * @date 2015年2月13日
 * @author salmon
 */

#ifndef CORE_GTL_ITERATOR_SP_NDARRAY_ITERATOR_H_
#define CORE_GTL_ITERATOR_SP_NDARRAY_ITERATOR_H_
#include "range.h"

namespace simpla
{

template<size_t NDIMS, typename IndexType = size_t>
struct sp_ndarray_iterator: public std::iterator<
		typename std::bidirectional_iterator_tag, nTuple<IndexType, NDIMS>,
		nTuple<IndexType, NDIMS> >
{
	static constexpr size_t ndims = NDIMS;

	typedef IndexType index_type;

	typedef nTuple<index_type, ndims> index_tuple;

private:
	index_tuple m_min_, m_max_, m_self_;
public:
	sp_ndarray_iterator()
	{
	}

	template<typename T1, typename T2>
	sp_ndarray_iterator(T1 const &min, T2 const &max)
	{
		m_min_ = min;
		m_max_ = max;
		m_self_ = min;
	}
	template<typename T1, typename T2, typename T3>
	sp_ndarray_iterator(T1 const &min, T2 const &max, T3 const &s)
	{
		m_min_ = min;
		m_max_ = max;
		m_self_ = s;
	}
	sp_ndarray_iterator(sp_ndarray_iterator const& other)
			: m_min_(other.m_min_), m_max_(other.m_max_), m_self_(other.m_self_)
	{

	}
	sp_ndarray_iterator(sp_ndarray_iterator && other)
			: m_min_(other.m_min_), m_max_(other.m_max_), m_self_(other.m_self_)
	{

	}
	~sp_ndarray_iterator()
	{
	}
	sp_ndarray_iterator & operator=(sp_ndarray_iterator const& other)
	{
		sp_ndarray_iterator(other).swap(*this);
		return *this;
	}
	void swap(sp_ndarray_iterator & other)
	{
		std::swap(m_min_, m_min_);
		std::swap(m_max_, m_max_);
		std::swap(m_self_, m_self_);

	}
	index_tuple const & operator *() const
	{
		return m_self_;
	}
	sp_ndarray_iterator & operator++()
	{
		int n = ndims - 1;

		++m_self_[n];

		while (n > 0 && m_self_[n] >= m_max_[n])
		{
			m_self_[n] = m_min_[n];
			++m_self_[n - 1];
			--n;
		}

		return *this;
	}

	sp_ndarray_iterator & operator--()
	{
		int n = ndims - 1;

		--m_self_[n];

		while (n > 0 && m_self_[n] >= m_max_[n])
		{
			m_self_[n] = m_min_[n];
			++m_self_[n - 1];
			--n;
		}

		return *this;
	}
	sp_ndarray_iterator operator++(int)
	{
		sp_ndarray_iterator res(*this);
		++(*this);
		return std::move(res);
	}
	sp_ndarray_iterator operator--(int)
	{
		sp_ndarray_iterator res(*this);
		--(*this);
		return std::move(res);
	}

	bool operator==(sp_ndarray_iterator const & other) const
	{
		return m_self_ == other.m_self_;
	}

	bool operator!=(sp_ndarray_iterator const & other) const
	{
		return m_self_ != other.m_self_;
	}

};
template<size_t NDIMS, typename IndexType>
constexpr size_t sp_ndarray_iterator<NDIMS, IndexType>::ndims;

template<size_t NDIMS, typename IndexType>
IndexType distance(sp_ndarray_iterator<NDIMS, IndexType> const & first,
		sp_ndarray_iterator<NDIMS, IndexType> const & second)
{
	return NProduct(*second - *first);

}
template<size_t NDIMS, typename IndexType = size_t>
struct sp_ndarray_range: public Range<sp_ndarray_iterator<NDIMS, IndexType> >
{
private:

public:
	static constexpr size_t array_order = C_ORDER;
	static constexpr size_t ndims = NDIMS;
	typedef sp_ndarray_iterator<NDIMS, IndexType> iterator_type;
	typedef Range<sp_ndarray_iterator<NDIMS, IndexType> > base_range;
	typedef nTuple<IndexType, NDIMS> index_tuple;

	index_tuple m_min_, m_max_;

	sp_ndarray_range()
	{
	}

	template<typename T1, typename T2>
	sp_ndarray_range(T1 const & min, T2 const & max)
			: base_range(iterator_type(min, max, min),
					(++iterator_type(min, max, max - 1)))
	{
		m_min_ = min;
		m_max_ = max;
	}
	sp_ndarray_range(sp_ndarray_range const & other)
			: base_range(other), m_max_(other.m_max_), m_min_(other.m_min_)
	{
	}
	sp_ndarray_range(sp_ndarray_range & other, op_split)
			: base_range(other, op_split()), m_max_(other.m_max_), m_min_(
					other.m_min_)
	{
	}
	sp_ndarray_range & operator=(sp_ndarray_range const & other)
	{
		sp_ndarray_range(other).swap(*this);
		return *this;
	}
	void swap(sp_ndarray_range & other)
	{
		base_range::swap(other);

		std::swap(m_max_, other.m_max_);
		std::swap(m_min_, other.m_min_);
	}
	size_t size() const
	{
		return NProduct(m_max_ - m_min_);
	}

};

}
// namespace simpla

#endif /* CORE_GTL_ITERATOR_SP_NDARRAY_ITERATOR_H_ */
