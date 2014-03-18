/*
 * field_vector_ops.h
 *
 *  Created on: 2014年3月11日
 *      Author: salmon
 */

#ifndef FIELD_VECTOR_OPS_H_
#define FIELD_VECTOR_OPS_H_
#include <cstddef>
#include <type_traits>

#include "constant_ops.h"
#include "field.h"
#include "primitives.h"
#include "ntuple.h"
#include "../utilities/type_utilites.h"

namespace simpla
{

//****************************************************************************************************
// For Vector Fields

//namespace fetl_impl
//{
//template<typename TM, typename TL, typename TR, typename TI> inline auto OpEval(Int2Type<DOT>, TM const &mesh,
//        Field<TM, VERTEX, TL> const &l, Field<TM, VERTEX, TR> const &r, TI s)
//        DECL_RET_TYPE((Dot(l.get(s) , r.get(s))))
////
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DOT>,
////		Field<TM, VERTEX, TL> const &l, nTuple<3, TR> const &r, TI s)
////		DECL_RET_TYPE((Dot(l.get(s) , r)))
////
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DOT>,
////		nTuple<3, TL> const & l, Field<TM, VERTEX, TR> const & r, TI s)
////		DECL_RET_TYPE((Dot(l , r.get(s))))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto OpEval(Int2Type<CROSS>, TM const &mesh,
//        Field<TM, VERTEX, TL> const &l, Field<TM, VERTEX, TR> const &r, TI s)
//        DECL_RET_TYPE((Cross(l.get(s) , r.get(s))))
//
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<CROSS>,
////		Field<TM, VERTEX, TL> const &l, nTuple<3, TR> const &r, TI s)
////		DECL_RET_TYPE((Cross(l.get(s) , r)))
////
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<CROSS>,
////		nTuple<3, TL> const & l, Field<TM, VERTEX, TR> const & r, TI s)
////		DECL_RET_TYPE((Cross(l , r.get(s))))
//
//}// namespace fetl_impl

//template<typename TG, typename TL, typename TR> inline auto //
//Dot(Field<TG, VERTEX, TL> const & lhs, nTuple<3, TR> const & rhs)
//DECL_RET_TYPE( (Field<TG,VERTEX , BiOp<DOT,Field<TG,VERTEX, TL> ,
//				nTuple<3, TR> > >(lhs, rhs)))
//
//template<typename TG, typename TL, typename TR> inline auto //
//Dot(nTuple<3, TL> const & lhs, Field<TG, VERTEX, TR> const & rhs)
//DECL_RET_TYPE( (Field<TG,VERTEX , BiOp<DOT,nTuple<3, TL> ,
//				Field<TG,VERTEX, TR> > >(lhs, rhs)))
//
//template<typename TG, typename TL, typename TR> inline auto //
//Cross(Field<TG, VERTEX, TL> const & lhs, nTuple<3, TR> const & rhs)
//DECL_RET_TYPE( (Field<TG,VERTEX , BiOp<CROSS,Field<TG,VERTEX, TL> ,
//				nTuple<3,TR> > >(lhs, rhs)))
//
//template<typename TG, typename TL, typename TR> inline auto //
//Cross(nTuple<3, TL> const & lhs, Field<TG, VERTEX, TR> const & rhs)
//DECL_RET_TYPE( (Field<TG,VERTEX , BiOp<CROSS,nTuple<3,TL> ,
//				Field<TG,VERTEX, TR> > >(lhs, rhs)))
////

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<TG, VERTEX, TL> const & lhs, Field<TG, VERTEX, TR> const & rhs)
DECL_RET_TYPE ((Field<TG, VERTEX, BiOp<DOT, Field<TG, VERTEX, TL>, Field<TG, VERTEX, TR> > >(lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<TG, VERTEX, TL> const & lhs, Field<TG, VERTEX, TR> const & rhs)
DECL_RET_TYPE ((Field<TG, VERTEX, BiOp<CROSS, Field<TG, VERTEX, TL>, Field<TG, VERTEX, TR> > >(lhs, rhs)))

template<typename TM, typename TL, typename TR>
struct Field<TM, VERTEX, BiOp<CROSS, TL, TR> >
{

public:
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr int IForm = VERTEX;
	static constexpr int TOP = CROSS;

	typedef Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::index_type index_type;

	mesh_type const & mesh;

	Field(TL const & l, TR const & r) :
			mesh(l.mesh), l_(l), r_(r)
	{
	}

	inline auto get(index_type s) const->decltype( Cross(l_.get(s) , r_.get(s)) )
	{
		return (Cross(l_.get(s), r_.get(s)));
	}

	inline auto operator[](index_type s) const
	DECL_RET_TYPE( (get(s)) )

}
;

template<typename TM, typename TL, typename TR>
struct Field<TM, VERTEX, BiOp<DOT, TL, TR> >
{

public:
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr int IForm = VERTEX;
	static constexpr int TOP = DOT;

	typedef Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::index_type index_type;

	mesh_type const & mesh;

	Field(TL const & l, TR const & r) :
			mesh(l.mesh), l_(l), r_(r)
	{
	}

	inline auto get(index_type s) const->decltype( Dot(l_.get(s) , r_.get(s)) )
	{
		return Dot(l_.get(s), r_.get(s));
	}

	inline auto operator[](index_type s) const
	DECL_RET_TYPE( (get(s)) )

}
;

}  // namespace simpla

#endif /* FIELD_VECTOR_OPS_H_ */
