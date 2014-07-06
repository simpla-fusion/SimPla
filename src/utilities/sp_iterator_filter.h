/*
 * sp_iterator_filter.h
 *
 *  Created on: 2014年6月5日
 *      Author: salmon
 */

#ifndef SP_ITERATOR_FILTER_H_
#define SP_ITERATOR_FILTER_H_

#include <map>
#include "sp_iterator.h"

namespace simpla
{
class _iterator_policy_filter
{
};

/**
 *  \ingroup iterator
 *
 *  \cite  boost::filter_iterator
 *
 */
template<typename TIterator, typename TPred>
struct Iterator<TIterator, TPred, _iterator_policy_filter, true>
{

	typedef TIterator base_iterator;

	typedef typename base_iterator::iterator_category iterator_category;
	typedef typename base_iterator::value_type value_type;
	typedef typename base_iterator::difference_type difference_type;
	typedef typename base_iterator::pointer pointer;
	typedef typename base_iterator::reference reference;

	typedef TPred predicate_fun;

	typedef Iterator<base_iterator, predicate_fun, _iterator_policy_filter, true> this_type;

	base_iterator it_;
	base_iterator it_end_;
	predicate_fun predicate_;

	Iterator()
	{
	}

	Iterator(this_type const & other)
			: it_(other.it_), it_end_(other.it_end_), predicate_(other.predicate_)
	{
	}

	Iterator(this_type && other)
			: it_(other.it_), it_end_(other.it_end_), predicate_(other.predicate_)
	{
	}

	Iterator(base_iterator const & ib, base_iterator ie, predicate_fun const &p)
			: it_(ib), it_end_(ie), predicate_(p)
	{
		while (it_ != (it_end_) && !(predicate_(*it_)))
		{
			++it_;
		}
	}

	~Iterator()
	{
	}
	bool operator ==(this_type const &other) const
	{
		return it_ == other.it_;
	}
	bool operator !=(this_type const &other) const
	{
		return it_ != other.it_;
	}

	this_type & operator ++()
	{
		while (it_ != (it_end_) && !(predicate_(*(++it_))))
		{
		}

		return *this;
	}
	this_type operator ++(int)
	{
		this_type res(*this);
		++res;
		return std::move(res);
	}
	this_type & operator --()
	{
		while (it_ != (it_end_) && !(predicate_(*(--it_))))
		{
		}

		return *this;
	}
	this_type operator --(int)
	{
		this_type res(*this);
		--res;
		return std::move(res);
	}
	reference operator *()
	{
		return *it_;
	}
	const reference operator *() const
	{
		return *it_;
	}

};
template<typename ...Args>
auto make_iterator_filter(Args&& ... args)
DECL_RET_TYPE((make_iterator(_iterator_policy_filter(),std::forward<Args>(args)...)))

template<typename ...Args>
auto make_range_filter(Args&& ... args)
DECL_RET_TYPE((make_range<_iterator_policy_filter>( std::forward<Args>(args)...)))

}
// namespace simpla

#endif /* SP_ITERATOR_FILTER_H_ */
