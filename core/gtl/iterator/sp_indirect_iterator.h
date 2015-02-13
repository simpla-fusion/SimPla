/**
 * @file sp_indirect_iterator.h
 *
 * @date 2015年2月13日
 * @author salmon
 */

#ifndef CORE_GTL_ITERATOR_SP_INDIRECT_ITERATOR_H_
#define CORE_GTL_ITERATOR_SP_INDIRECT_ITERATOR_H_

#include "../containers/container_traits.h"
namespace simpla
{

template<typename BaseIterator, typename Container>
struct sp_indirect_iterator: public BaseIterator
{
	typedef typename std::iterator_traits<BaseIterator>::value_type key_type;
	typedef Container container_type;
	typedef BaseIterator base_iterator_type;
	typedef typename container_traits<container_type>::value_type value_type;

	typedef sp_indirect_iterator<base_iterator_type, container_type> this_type;

	container_type & m_container_;

	sp_indirect_iterator(base_iterator_type const & base_it,
			container_type & container)
			: base_iterator_type(base_it), m_container_(container)
	{
	}
	sp_indirect_iterator(this_type const & other)
			: base_iterator_type(other), m_container_(other.m_container_)
	{

	}
	~sp_indirect_iterator()
	{
	}
	using base_iterator_type::operator++;

	value_type & operator*()
	{
		return m_container_[base_iterator_type::operator*()];
	}
	value_type & operator->()
	{
		return m_container_[base_iterator_type::operator*()];
	}

	sp_indirect_iterator operator++(int) const
	{
		this_type res(*this);
		++res;
		return std::move(res);
	}

	bool operator==(sp_indirect_iterator const & other) const
	{
		return base_iterator_type::operator==(other)
				&& (&m_container_ == &other.m_container_);
	}

	bool operator!=(sp_indirect_iterator const & other) const
	{
		return base_iterator_type::operator!=(other)
				|| (&m_container_ != &other.m_container_);
	}

};

template<typename BaseIterator, typename Container>
sp_indirect_iterator<BaseIterator, Container> indirect_iterator(
		BaseIterator const & first, Container & container)
{
	return sp_indirect_iterator<BaseIterator, Container>(first, container);
}

template<typename KeyRange, typename ValueContainer>
struct sp_indirect_range
{
	typedef KeyRange key_range_type;
	typedef ValueContainer value_container_type;

	key_range_type m_key_range_;
	value_container_type & m_value_;

	typedef typename key_range_type::iterator key_iterator;

	typedef sp_indirect_iterator<key_iterator, value_container_type> iterator;
	typedef sp_indirect_iterator<key_iterator, const value_container_type> const_iterator;

	sp_indirect_range(key_range_type && key_range,
			value_container_type & values)
			: m_key_range_(key_range), m_value_(values)
	{
	}

	sp_indirect_range(sp_indirect_range const & other)
			: m_key_range_(other.m_key_range_), m_value_(other.m_value_)
	{
	}

	~sp_indirect_range()
	{
	}

	//! @name TBB::Range Concept
	//! @{

	template<typename ...Others>
	sp_indirect_range(sp_indirect_range & other, Others &&...others)
			: m_key_range_(other.m_key_range_, std::forward<Others>(others)...), m_value_(
					other.m_value_)
	{
	}
	constexpr bool empty() const
	{
		return m_key_range_.empty();
	}
	constexpr bool is_divisible() const
	{
		return m_key_range_.is_divisible();
	}
	static const bool is_splittable_in_proportion()
	{
		return key_range_type::is_splittable_in_proportion();
	}

	//! @}

	iterator begin()
	{
		return indirect_iterator(m_key_range_.begin(), m_value_);
	}
	iterator end()
	{
		return indirect_iterator(m_key_range_.end(), m_value_);
	}

	const_iterator begin() const
	{
		return indirect_iterator(m_key_range_.begin(), m_value_);
	}

	const_iterator end() const
	{
		return indirect_iterator(m_key_range_.end(), m_value_);
	}
	const_iterator cbegin() const
	{
		return indirect_iterator(m_key_range_.begin(), m_value_);
	}

	const_iterator cend() const
	{
		return indirect_iterator(m_key_range_.end(), m_value_);
	}
};
}  // namespace simpla

#endif /* CORE_GTL_ITERATOR_SP_INDIRECT_ITERATOR_H_ */
