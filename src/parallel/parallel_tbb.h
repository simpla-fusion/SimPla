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

}

template<typename Range, typename Value, typename RealBody, typename Reduction, typename ... Others>
Value parallel_reduce(const Range& range, const Value& identity, const RealBody& real_body, const Reduction& reduction,
        Others&&... others)
{
	return identity;
}
template<typename Range, typename Function, typename ... Others>
void parallel_for_each(Range& rng, const Function& f, Others &&...others)
{
}
}  // namespace simpla

#endif /* PARALLEL_TBB_H_ */
