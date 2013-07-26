/*
 * operator_overload.h
 *
 *  Created on: 2013-7-25
 *      Author: salmon
 */

#ifndef OPERATOR_OVERLOAD_H_
#define OPERATOR_OVERLOAD_H_
#include "expression.h"

namespace simpla
{
//
//template<typename TE> inline TE const &
//operator +(TE const &e, Zero const &)
//{
//	return (e);
//}
//
//template<typename TE> inline TE const &
//operator +(Zero const &, TE const &e)
//{
//	return (e);
//}
//
//template<typename TE> inline TE const &
//operator -(TE const &e, Zero const &)
//{
//	return (e);
//}
//
//template<typename TE> inline auto operator -(Zero const &, TE const &e)
//DECL_RET_TYPE((Negate(e)))
//
//Zero operator +(Zero const &, Zero const &e)
//{
//	return (Zero());
//}
//
//template<typename TE> inline TE const &operator *(TE const &e, One const &)
//{
//	return (e);
//}
//
//template<typename TE> inline TE const & operator *(One const &, TE const &e)
//{
//	return (e);
//}
//
//template<typename TE> inline Zero operator *(TE const &, Zero const &)
//{
//	return (Zero());
//}
//
//template<typename TE> inline Zero operator *(Zero const &, TE const &)
//{
//	return (Zero());
//}
//
//template<typename TE> inline Infinity operator /(TE const &e, Zero const &)
//{
//	return (Infinity());
//}
//
//template<typename TE> inline Zero operator /(Zero const &, TE const &e)
//{
//	return (Zero());
//}
//
//template<typename TE> inline Zero operator /(TE const &, Infinity const &)
//{
//	return (Zero());
//}
//
//template<typename TE> inline Infinity operator /(Infinity const &, TE const &e)
//{
//	return (Infinity());
//}

template<typename T> inline auto  //
operator-(T const & lhs)
DECL_RET_TYPE((Negate(lhs)))

template<typename TL, typename TR> inline auto //
operator +(TL const &lhs, TR const & rhs)
DECL_RET_TYPE(Plus(lhs,rhs))

template<typename TL, typename TR> inline auto //
operator -(TL const &lhs, TR const & rhs)
DECL_RET_TYPE(Minus(lhs,rhs))

template<typename TL, typename TR> inline auto //
operator *(TL const &lhs, TR const & rhs)
DECL_RET_TYPE(Multiplies(lhs,rhs))

template<typename TL, typename TR> inline auto //
operator /(TL const &lhs, TR const & rhs)
DECL_RET_TYPE(Divides(lhs,rhs))

}  // namespace simpla

#endif /* OPERATOR_OVERLOAD_H_ */
