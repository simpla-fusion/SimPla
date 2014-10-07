/*
 * parallel_tbb.h
 *
 *  Created on: 2014年9月4日
 *      Author: salmon
 */

#ifndef PARALLEL_TBB_H_
#define PARALLEL_TBB_H_

#include "tbb/tbb.h"
namespace simpla
{

//using tbb interface

template<typename Range, typename Func>
void parallel_for(Range const & range, Func const & fun)
{
	for (auto const& s : range)
	{
		f(s);
	}
}

template<typename Range, typename Value, typename OP, typename Reduction,
		typename ... Args>
Value parallel_reduce(const Range& range, const OP& op, const Reduction& reduce,
		Args&&... args)
{
	auto b = begin(range);
	auto e = end(range);

	auto res = op(get_value(std::forward<Args>(args),b)...);
	++b;

	for (; b != e; ++b)
	{
		reduce(res, op(get_value(std::forward<Args>(args),b)...));
	}
	return res;
}
template<typename Range, typename Function, typename ... Others>
void parallel_for_each(Range& range, const Function& f, Others &&...others)
{
	for (auto const& s : range)
	{
		f(get_value(std::forward<Others>(others),s)...);
	}
}
}  // namespace simpla

#endif /* PARALLEL_TBB_H_ */
