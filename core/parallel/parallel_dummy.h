/*
 * parallel_dummy.h
 *
 *  Created on: 2014年11月6日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_PARALLEL_DUMMY_H_
#define CORE_PARALLEL_PARALLEL_DUMMY_H_

#include "../utilities/sp_foreach.h"
namespace simpla
{

/**
 *
 * @param range Range Concept
 * @param op std::function<void(Range)>
 */
template<typename Range, typename OP>
void parallel_for(Range const & range, OP const & op)
{
	op(range);
}

/**
 *
 * @param range Range Concept
 * @param op std::function<void(*Range::iterator)>
 */
template<typename Range, typename OP>
void parallel_foreach(Range const & range, OP const & op)
{
	sp_foreach(range, op);
}

/**
 *
 * @param range  Range Concept
 * @param op     std::function<T(Range)>
 * @param reduce std::function<T(T,T)>
 * @return T
 */
template<typename Value, typename Range, typename OP, typename Reduction>
auto parallel_reduce(const Range& range, OP const& op,
		const Reduction& reduce)->
		typename std::result_of<OP(Range const &)>::type
{
	return op(range);
}
/**
 *
 * @param range  Range Concept
 * @param op     std::function<T(Range)>
 * @return T
 */
template<typename Value, typename Range, typename OP>
auto parallel_reduce(const Range& range, OP const& op)->
typename std::result_of<OP(Range const &)>::type
{

	typedef typename std::result_of<OP(Range const &)>::type res_type;

	return parallel_reduce(range, op, std::plus<res_type>());
}

} // namespace simpla

#endif /* CORE_PARALLEL_PARALLEL_DUMMY_H_ */
