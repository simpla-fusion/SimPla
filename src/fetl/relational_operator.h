/*
 * relational_operator.h
 *
 *  Created on: 2013-7-9
 *      Author: salmon
 */

#ifndef RELATIONAL_OPERATOR_H_
#define RELATIONAL_OPERATOR_H_
#include "fetl_defs.h"

namespace simpla
{

template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
Field<TG, IFORM,
		_impl::OpEquality<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> > >  //
operator==(Field<TG, IFORM, TLExpr> const & lhs,
		Field<TG, IFORM, TRExpr> const & rhs)
{
	return (Field<TG, IFORM,
			_impl::OpEquality<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> > >(
			lhs, rhs));
}




}  // namespace simpla

#endif /* RELATIONAL_OPERATOR_H_ */
