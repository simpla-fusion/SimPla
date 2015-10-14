/*
 * sp_range_filter.h
 *
 *  Created on: 2014-10-29
 *      Author: salmon
 */

#ifndef CORE_CONTAINERS_SP_RANGE_FILTER_H_
#define CORE_CONTAINERS_SP_RANGE_FILTER_H_

#include "../containers/sp_iterator_filter.h"
namespace simpla
{

template<typename ...> struct FilterRange;

template<typename BaseRange>
struct FilterRange<BaseRange> : public BaseRange
{
	typedef BaseRange base_range;

	typedef typename base_range::iterator base_iterator;

	typedef decltype(*std::declval<base_iterator>()) value_type;

	typedef std::function<bool(value_type const &)> pred_function;

	typedef Iterator<base_iterator, pred_function, _iterator_policy_filter, true> iterator;

	pred_function pred_;

	template<typename TPredFun>
	FilterRange(base_range const & r, TPredFun const & pred) :
			base_range(r), pred_(pred)
	{

	}
	~FilterRange()
	{
	}

	iterator begin()
	{
		return std::move(
				iterator(base_range::begin(), base_range::end(), pred_));
	}

	iterator end()
	{
		return std::move(iterator(base_range::end(), base_range::end(), pred_));
	}

	iterator begin() const
	{
		return std::move(
				iterator(base_range::begin(), base_range::end(), pred_));
	}

	iterator end() const
	{
		return std::move(iterator(base_range::end(), base_range::end(), pred_));
	}

	iterator rbegin()
	{
		return std::move(
				iterator(base_range::rbegin(), base_range::rend(), pred_));
	}

	iterator rend()
	{
		return std::move(
				iterator(base_range::rend(), base_range::rend(), pred_));
	}

	iterator rbegin() const
	{
		return std::move(
				iterator(base_range::rbegin(), base_range::rend(), pred_));
	}

	iterator rend() const
	{
		return std::move(
				iterator(base_range::rend(), base_range::rend(), pred_));
	}
};
}
// namespace simpCORE_CONTAINERS_SP_RANGE_FILTER_H_ANGE_FILTER_H_ */
