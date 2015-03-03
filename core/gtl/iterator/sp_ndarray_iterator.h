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

	template<typename TIndices>
	sp_ndarray_iterator(TIndices const &min, TIndices const &max)
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
	index_tuple m_min_, m_max_, m_strides_;
	size_t m_max_hash_ = 0;

	sp_ndarray_range()
	{
	}

	sp_ndarray_range(index_tuple min, index_tuple max)
			: base_range(iterator_type(min, max, min),
					(++iterator_type(min, max, max - 1))), m_min_(min), m_max_(
					max)
	{
		m_strides_[ndims - 1] = 1;
		if (ndims > 1)
		{
			for (int i = ndims - 2; i >= 0; --i)
			{
				m_strides_[i] = (m_max_[i + 1] - m_min_[i + 1])
						* m_strides_[i + 1];
			}
		}
		m_max_hash_ = m_strides_[0] * (m_max_[0] - m_min_[0]);
	}
	sp_ndarray_range(sp_ndarray_range const & other)
			: base_range(other), m_max_(other.m_max_), m_min_(other.m_min_), m_strides_(
					other.m_strides_), m_max_hash_(other.m_max_hash_)
	{
	}
	sp_ndarray_range(sp_ndarray_range & other, op_split)
			: base_range(other, op_split()), m_max_(other.m_max_), m_min_(
					other.m_min_), m_strides_(other.m_strides_), m_max_hash_(
					other.m_max_hash_)
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
		std::swap(m_strides_, other.m_strides_);
		std::swap(m_max_hash_, other.m_max_hash_);
	}
	size_t size() const
	{
		return NProduct(m_max_ - m_min_);
	}
	size_t max_hash() const
	{
		return NProduct(m_max_ - m_min_);
	}
	size_t hash(index_tuple const & s) const
	{
		return inner_product(s - m_min_, m_strides_);
	}

	size_t hash(size_t N0, size_t N1) const
	{
		return (N0 - m_min_[0]) * m_strides_[0]
				+ (N1 - m_min_[1]) * m_strides_[1];
	}

	size_t hash(size_t N0, size_t N1, size_t N2) const
	{
		return (N0 - m_min_[0]) * m_strides_[0]
				+ (N1 - m_min_[1]) * m_strides_[1]
				+ (N2 - m_min_[2]) * m_strides_[2];
	}

};
}
// namespace simpla

#endif /* CORE_GTL_ITERATOR_SP_NDARRAY_ITERATOR_H_ */
