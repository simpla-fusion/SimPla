/*
 * constant_ops.h
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#ifndef CONSTANT_OPS_H_
#define CONSTANT_OPS_H_

#include "primitives.h"

namespace simpla
{

template<typename value_type>
struct Constant
{
	value_type value;
};

struct Zero
{
};

struct One
{
};
struct Infinity
{
};

struct Undefine
{
};

template<typename TE> inline TE const &
operator +(TE const &e, Zero const &)
{
	return (e);
}

template<typename TE> inline TE const &
operator +(Zero const &, TE const &e)
{
	return (e);
}

template<typename TE> inline TE const &
operator -(TE const &e, Zero const &)
{
	return (e);
}

//template<typename TE> inline auto operator -(Zero const &, TE const &e)
//DECL_RET_TYPE (((-e)))

inline Zero operator +(Zero const &, Zero const &e)
{
	return (Zero());
}

template<typename TE> inline TE const &operator *(TE const &e, One const &)
{
	return (e);
}

template<typename TE> inline TE const & operator *(One const &, TE const &e)
{
	return (e);
}

template<typename TE> inline Zero operator *(TE const &, Zero const &)
{
	return (Zero());
}

template<typename TE> inline Zero operator *(Zero const &, TE const &)
{
	return (Zero());
}

template<typename TE> inline Infinity operator /(TE const &e, Zero const &)
{
	return (Infinity());
}

template<typename TE> inline Zero operator /(Zero const &, TE const &e)
{
	return (Zero());
}

template<typename TE> inline Zero operator /(TE const &, Infinity const &)
{
	return (Zero());
}

template<typename TE> inline Infinity operator /(Infinity const &, TE const &e)
{
	return (Infinity());
}

template<typename TL> inline auto   //
operator==(TL const & lhs, Zero)
DECL_RET_TYPE ((lhs))

template<typename TR> inline auto   //
operator==(Zero, TR const & rhs)
DECL_RET_TYPE ((rhs))
}  // namespace simpla

#endif /* CONSTANT_OPS_H_ */
