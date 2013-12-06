/*
 * mesh.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef MESH_H_
#define MESH_H_

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <type_traits>
#include "fetl/primitives.h"

namespace simpla
{

template<typename, int> class Geometry;
template<typename, typename > class Field;

template<typename TM, int TOP, typename TL, typename TR = std::nullptr_t>
struct mesh_has_op
{
	template<typename T1, typename T2, typename T3>
	static auto check_op(Int2Type<TOP>, T1 const& m, T2 const &l, T3 const &r)
	->decltype(m.template OpEval(Int2Type<TOP>(),l,r,0) )
	{
	}

	static std::false_type check_op(...)
	{
		return std::false_type();
	}
	typedef decltype(
			check_op(Int2Type<TOP>(),std::declval<TM>(),
					std::declval<TL>(),
					std::declval<TR>())
	) result_type;

public:

	static const bool value =
			!(std::is_same<result_type, std::false_type>::value);

}
;
template<typename TM, int TOP, typename TL>
struct mesh_has_op<TM, TOP, TL, std::nullptr_t>
{
	template<typename T1, typename T2>
	static auto check_op(Int2Type<TOP>, T1 const& m, T2 const &l)
	->decltype(m.template OpEval(Int2Type<TOP>(),l) )
	{
	}

	static std::false_type check_op(...)
	{
		return std::false_type();
	}
	typedef decltype( check_op (Int2Type<TOP>(),std::declval<TM>(), std::declval<TL>()) ) result_type;

public:

	static const bool value =
			!(std::is_same<result_type, std::false_type>::value);

};


}
//namespace simpla

#endif /* MESH_H_ */
