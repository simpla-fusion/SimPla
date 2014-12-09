/*
 * parallel_dummy.h
 *
 *  Created on: 2014年11月6日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_PARALLEL_DUMMY_H_
#define CORE_PARALLEL_PARALLEL_DUMMY_H_

namespace simpla
{
template<typename Range, typename OP>
void parallel_for(Range const & range, OP const & op)
{
	op(range);
}
template<typename Range, typename OP>
void parallel_foreach(Range const & range, OP const & op)
{
	for (auto const & s : range)
	{
		op(s);
	}
}
template<typename Value, typename Range, typename OP, typename Reduction>
auto parallel_reduce(const Range& range, const OP& op,
		const Reduction& reduce)->
		typename std::result_of<OP(Range const &)>::type
{
//	auto b = begin(range);
//	auto e = end(range);
//
//	auto res = op(get_value(std::forward<Args>(args),*b)...);
//	++b;
//
//	for (; b != e; ++b)
//	{
//		reduce(res, op(get_value(std::forward<Args>(args),*b)...));
//	}
	return op(range);
}

//template<typename Range, typename Reduction, typename Args, typename Value>
//Value parallel_reduce(const Range& range, const Reduction& reduce, Args&& args)
//{
//	auto b = begin(range);
//	auto e = end(range);
//
//	auto res = get_value(std::forward<Args>(args), *b);
//	++b;
//
//	for (; b != e; ++b)
//	{
//		res = reduce(res, get_value(std::forward<Args>(args), *b));
//	}
//	return res;
//}
//template<typename Range, typename Function, typename ... Others>
//void parallel_for_each(Range& range, const Function& f, Others &&...others)
//{
//	for (auto const& s : range)
//	{
//		f(get_value(std::forward<Others>(others),*s)...);
//	}
//}
}// namespace simpla

#endif /* CORE_PARALLEL_PARALLEL_DUMMY_H_ */
