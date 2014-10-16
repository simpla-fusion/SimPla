/*
 * parallel.h
 *
 *  created on: 2014-3-27
 *      Author: salmon
 */

#ifndef PARALLEL_H_
#define PARALLEL_H_

#include "multi_thread.h"

/**
 *  \defgroup  Parallel Parallel
 *  @{
 *  	\defgroup  MPI MPI Communicaion
 *  	\defgroup  MULTICORE Multi-thread/core and many-core support
 *  @}
 */

namespace simpla
{

template<typename Range, typename OP>
void parallel_for(Range const & range, OP const & op)
{
	for (auto const& s : range)
	{
		op(s);
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

template<typename Value, typename Range, typename Reduction, typename Args>
Value parallel_reduce(const Range& range, const Reduction& reduce, Args&& args)
{
	auto b = begin(range);
	auto e = end(range);

	auto res = get_value(std::forward<Args>(args), b);
	++b;

	for (; b != e; ++b)
	{
		res = reduce(res, get_value(std::forward<Args>(args), b));
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

#endif /* PARALLEL_H_ */
