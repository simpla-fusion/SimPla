/**
 * @file sp_ntuple_range.h
 *
 * @date 2015年2月13日
 * @author salmon
 */

#ifndef CORE_GTL_ITERATOR_SP_NTUPLE_RANGE_H_
#define CORE_GTL_ITERATOR_SP_NTUPLE_RANGE_H_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "../ntuple.h"
#include "../../numeric/geometric_algorithm.h"
#include "range.h"

namespace simpla
{
namespace _impl
{
template<typename T0, typename T1, typename TV, size_t NDIMS>
typename std::make_signed<TV>::type ntuple_carry(T0 const & min, T1 const & max,
		nTuple<TV, NDIMS> * self, typename std::make_signed<TV>::type flag = 1,
		bool C_ORDER = false)
{
	typedef TV value_type;

	typedef typename std::make_signed<TV>::type signed_value_type;

	for (int n = (C_ORDER) ? NDIMS - 1 : 0, ne = (C_ORDER) ? -1 : NDIMS, ni =
			(C_ORDER) ? -1 : 1; n != ne && flag != 0; n += ni)
	{

		auto div = std::div(
				static_cast<signed_value_type>((*self)[n]) + flag
						+ static_cast<signed_value_type>(max[n])
						- static_cast<signed_value_type>(min[n]) * 2,
				static_cast<signed_value_type>(max[n])
						- static_cast<signed_value_type>(min[n]));

		flag = div.quot - 1;

		(*self)[n] = static_cast<value_type>(div.rem)
				+ static_cast<value_type>(min[n]);

	}

	return flag;
}
template<typename T0, typename T1, typename TV, size_t NDIMS>
void ntuple_cycle(T0 const & min, T1 const & max, nTuple<TV, NDIMS> * self)
{
	typedef TV value_type;
	typedef typename std::make_signed<TV>::type signed_value_type;

	for (int n = NDIMS - 1; n >= 0; --n)
	{

		(*self)[n] = (

		static_cast<signed_value_type>((*self)[n])

		+ static_cast<signed_value_type>(max[n])

		- static_cast<signed_value_type>(min[n]) * 2

		) % (

		static_cast<signed_value_type>(max[n])

		- static_cast<signed_value_type>(min[n]))

		+ static_cast<value_type>(min[n]);
	}

}
}
// namespace _impl
template<typename IndexType, size_t ...DIMS>
struct sp_nTuple_range
{
public:

	typedef IndexType index_type;

	typedef sp_nTuple_range<IndexType, DIMS...> this_type;

	typedef nTuple<IndexType, DIMS...> ntuple_type;
	typedef nTuple<IndexType, DIMS...> value_type;
	struct iterator;
	typedef iterator const_iterator;

protected:

	ntuple_type m_b_, m_e_;

	ntuple_type m_grainsize_;

	bool m_slow_first_ = true;

public:
	sp_nTuple_range()
	{

	}
	template<typename T1, typename T2>
	sp_nTuple_range(T1 const & min, T2 const & max, bool slow_first = true) :
			m_slow_first_(slow_first)
	{
		m_b_ = min;
		m_e_ = max;
		m_grainsize_ = 1;
	}

	template<typename T1, typename T2, typename T3>
	sp_nTuple_range(T1 const & min, T2 const & max, T3 const & grain_size,
			bool slow_first = true) :
			m_slow_first_(slow_first)
	{
		m_b_ = min;
		m_e_ = max;
		m_grainsize_ = grain_size;
	}
	sp_nTuple_range(this_type & other, op_split) :
			m_grainsize_(other.m_grainsize_), m_slow_first_(other.m_slow_first_)
	{

		// @FIXME only valid when sizeof...(DIMS)=1
		//		m_b_ = (other.m_e_ - other.m_b_) / 2 + other.m_b_;
		//		m_e_ = other.m_e_;
		//		other.m_e_ = m_b_;
	}

	sp_nTuple_range(this_type const & other) :
			m_e_(other.m_e_), m_b_(other.m_b_), m_slow_first_(
					other.m_slow_first_), m_grainsize_(other.m_grainsize_)
	{
	}

	this_type & operator=(this_type const & other)
	{
		this_type(other).swap(*this);
		return *this;
	}
	void swap(this_type & other)
	{
		std::swap(m_b_, other.m_b_);
		std::swap(m_e_, other.m_e_);
		std::swap(m_grainsize_, other.m_grainsize_);
		std::swap(m_slow_first_, other.m_slow_first_);
	}

	size_t size() const
	{
		return NProduct(m_e_ - m_b_);
	}
	bool is_divisable() const
	{
		return false;
	}
	bool is_slow_first() const
	{
		return m_slow_first_;
	}

	void is_slow_first(bool slow_first)
	{
		m_slow_first_ = slow_first;
	}
	constexpr const_iterator begin() const
	{
		return (const_iterator(m_b_, m_e_, m_b_, m_slow_first_));
	}

	const_iterator end() const
	{
		const_iterator res(m_b_, m_e_, m_e_ - 1, m_slow_first_);
		++res;
		return std::move(res);
	}

	this_type operator&(this_type const& other) const
	{
		this_type res(*this);

		rectangle_overlap(other.m_b_, other.m_e_, &res.m_b_, &res.m_e_);

		return std::move(res);
	}
};
template<typename TIndex, size_t ... DIMS, typename T2>
sp_nTuple_range<TIndex, DIMS...> make_ntuple_range(
		nTuple<TIndex, DIMS...> const & b, T2 const & e, bool slow_first = true)
{
	return sp_nTuple_range<TIndex, DIMS...>(b, e, slow_first);
}

template<typename IndexType, size_t ... DIMS>
struct sp_nTuple_range<IndexType, DIMS...>::iterator: public std::iterator<
		typename std::bidirectional_iterator_tag, nTuple<IndexType, DIMS...>,
		nTuple<IndexType, DIMS...> >
{
public:

	typedef iterator this_type;

	typename std::make_signed<IndexType>::type m_sign_flag_ = 0;

	bool m_slow_first_ = true;
	ntuple_type m_min_, m_max_, m_self_;
public:

	template<typename T1, typename T2, typename T3>
	iterator(T1 const &min, T2 const &max, T3 const &s, bool slow_first = true)
	{
		m_min_ = min;
		m_max_ = max;
		m_self_ = s;
		m_slow_first_ = slow_first;
	}
	iterator(iterator const& other) :
			m_min_(other.m_min_), m_max_(other.m_max_), m_self_(other.m_self_), m_sign_flag_(
					other.m_sign_flag_), m_slow_first_(other.m_slow_first_)
	{
	}
	iterator(iterator && other) :
			m_min_(other.m_min_), m_max_(other.m_max_), m_self_(other.m_self_), m_sign_flag_(
					other.m_sign_flag_), m_slow_first_(other.m_slow_first_)
	{
	}
	~iterator()
	{
	}

	iterator & operator=(iterator const& other)
	{
		iterator(other).swap(*this);
		return *this;
	}
	void swap(iterator & other)
	{
		std::swap(m_min_, other.m_min_);
		std::swap(m_max_, other.m_max_);
		std::swap(m_self_, other.m_self_);
		std::swap(m_sign_flag_, other.m_sign_flag_);
		std::swap(m_slow_first_, other.m_slow_first_);

	}
	bool operator==(this_type const & other) const
	{
		return m_sign_flag_ == other.m_sign_flag_ && m_self_ == other.m_self_
				&& (m_min_ == other.m_min_) && (m_max_ == other.m_max_);
	}
	bool operator!=(this_type const & other) const
	{
		return m_sign_flag_ != other.m_sign_flag_ || m_self_ != other.m_self_
				|| (m_min_ != other.m_min_) || (m_max_ != other.m_max_);
	}

	ntuple_type const & operator *() const
	{
		return m_self_;
	}

	this_type & operator++()
	{

		m_sign_flag_ = _impl::ntuple_carry(m_min_, m_max_, &m_self_, 1,
				m_slow_first_);
		return *this;
	}

	this_type & operator--()
	{
		m_sign_flag_ = _impl::ntuple_carry(m_min_, m_max_, &m_self_, -1,
				m_slow_first_);

		return *this;
	}
	this_type operator++(int)
	{
		this_type res(*this);
		++(*this);
		return std::move(res);
	}
	this_type operator--(int)
	{
		this_type res(*this);
		--(*this);
		return std::move(res);
	}

};

template<size_t NDIMS, typename IndexType>
IndexType distance(
		typename sp_nTuple_range<IndexType, NDIMS>::iterator const & first,
		typename sp_nTuple_range<IndexType, NDIMS>::iterator const & second)
{
	return NProduct(*second - *first);

}
}
// namespace simpla

#endif /* CORE_GTL_ITERATOR_SP_NTUPLE_RANGE_H_ */
