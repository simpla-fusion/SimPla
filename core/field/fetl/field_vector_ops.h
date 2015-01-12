/*
 * field_vector_ops.h
 *
 *  created on: 2014-3-11
 *      Author: salmon
 */

#ifndef FIELD_VECTOR_OPS_H_
#define FIELD_VECTOR_OPS_H_

#include "field.h"

#include "../utilities/constant_ops.h"
#include "../utilities/primitives.h"
#include "../gtl/ntuple.h"
#include "../utilities/type_traits.h"

namespace simpla
{

//****************************************************************************************************
// For Vector Fields

//namespace fetl_impl
//{
//template<typename TM, typename TL, typename TR, typename TI> inline auto Opcalculus(std::integral_constant<unsigned int ,DOT>, TM const &mesh,
//        Field<TM, VERTEX, TL> const &l, Field<TM, VERTEX, TR> const &r, TI s)
//        DECL_RET_TYPE((Dot(l.get(s) , r.get(s))))
////
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpcalculus(std::integral_constant<unsigned int ,DOT>,
////		Field<TM, VERTEX, TL> const &l, nTuple<3, TR> const &r, TI s)
////		DECL_RET_TYPE((Dot(l.get(s) , r)))
////
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpcalculus(std::integral_constant<unsigned int ,DOT>,
////		nTuple<3, TL> const & l, Field<TM, VERTEX, TR> const & r, TI s)
////		DECL_RET_TYPE((Dot(l , r.get(s))))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto Opcalculus(std::integral_constant<unsigned int ,CROSS>, TM const &mesh,
//        Field<TM, VERTEX, TL> const &l, Field<TM, VERTEX, TR> const &r, TI s)
//        DECL_RET_TYPE((Cross(l.get(s) , r.get(s))))
//
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpcalculus(std::integral_constant<unsigned int ,CROSS>,
////		Field<TM, VERTEX, TL> const &l, nTuple<3, TR> const &r, TI s)
////		DECL_RET_TYPE((Cross(l.get(s) , r)))
////
////template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpcalculus(std::integral_constant<unsigned int ,CROSS>,
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
struct _Field<TM, VERTEX, BiOp<CROSS, TL, TR> >
{

public:
	typename reference_traits<TL>::const_reference l_;
	typename reference_traits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr   unsigned int   IForm = VERTEX;
	static constexpr   unsigned int   TOP = CROSS;

	typedef _Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::iterator iterator;

	mesh_type const & mesh;

	_Field(TL const & l, TR const & r)
			: mesh(l.mesh), l_(l), r_(r)
	{
	}
	template<typename TI>
	inline auto get(TI s) const->decltype( Cross(l_.get(s) , r_.get(s)) )
	{
		return (Cross(l_.get(s), r_.get(s)));
	}
	template<typename TI>
	inline auto operator[](TI s) const
	DECL_RET_TYPE( (get(s)) )

}
;

template<typename TM, typename TL, typename TR>
struct _Field<TM, VERTEX, BiOp<DOT, TL, TR> >
{

public:
	typename reference_traits<TL>::const_reference l_;
	typename reference_traits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr   unsigned int   IForm = VERTEX;
	static constexpr   unsigned int   TOP = DOT;

	typedef _Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::iterator iterator;

	mesh_type const & mesh;

	_Field(TL const & l, TR const & r)
			: mesh(l.mesh), l_(l), r_(r)
	{
	}
	template<typename TI>
	inline auto get(TI s) const->decltype( Dot(l_.get(s) , r_.get(s)) )
	{
		return Dot(l_.get(s), r_.get(s));
	}
	template<typename TI>
	inline auto operator[](TI s) const
	DECL_RET_TYPE( (get(s)) )

}
;

}  // namespace simpla

#endif /* FIELD_VECTOR_OPS_H_ */
