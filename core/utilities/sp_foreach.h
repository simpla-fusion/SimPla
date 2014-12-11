/*
 * sp_foreach.h
 *
 *  Created on: 2014年12月10日
 *      Author: salmon
 */

#ifndef CORE_UTILITIES_SP_FOREACH_H_
#define CORE_UTILITIES_SP_FOREACH_H_
#include "../utilities/sp_type_traits.h"

namespace simpla
{
HAS_MEMBER_FUNCTION(foreach)

HAS_MEMBER_FUNCTION(reduce)
/**
 *
 * @param range Range Concept
 * @param op std::function<void(*Range::iterator)>
 */
template<typename Range, typename OP>
typename std::enable_if<has_member_function_foreach<Range, OP>::value, void>::type sp_foreach(
		Range const & range, OP const & op)
{
	range.foreach(op);
}

template<typename Range, typename OP>
typename std::enable_if<!has_member_function_foreach<Range, OP>::value, void>::type sp_foreach(
		Range const & range, OP const & op)
{
	for (auto const & s : range)
	{
		op(s);
	}
}

}  // namespace simpla

#endif /* CORE_UTILITIES_SP_FOREACH_H_ */
