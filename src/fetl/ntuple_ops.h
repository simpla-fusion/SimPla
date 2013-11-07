/*
 * ntuple_ops.h
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#ifndef NTUPLE_OPS_H_
#define NTUPLE_OPS_H_

#include <fetl/primitives.h>

namespace simpla
{
// Expression template of nTuple

template<int N, typename TL, typename TR> inline auto _OpPLUS(
		nTuple<N, TL> const & lhs,
		nTuple<N, TR> const & rhs)
				DECL_RET_TYPE(((nTuple<N, BiOp<PLUS ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))

}
// namespace simpla

#endif /* NTUPLE_OPS_H_ */
