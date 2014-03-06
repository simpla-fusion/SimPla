/*
 *  operations.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include <cstddef>
#include <type_traits>

#include "../mesh/mesh.h"
#include "constant_ops.h"
#include "field.h"
#include "primitives.h"
#include "ntuple.h"
#include "../utilities/type_utilites.h"

namespace simpla
{

template<int, typename > class nTuple;
//****************************************************************************************************
template<typename TM, int IL, typename TL, typename TR> inline auto //
operator==(Field<TM, IL, TL> const & lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((lhs-rhs))
//****************************************************************************************************

namespace fetl_impl
{

template<typename TM, int IL, typename TL, typename TI> inline auto FieldOpEval(Int2Type<NEGATE>,
		Field<TM, IL, TL> const & f, TI s)
		DECL_RET_TYPE((-f.get(s)) )

template<typename TM, int IL, typename TL, typename TI> inline auto FieldOpEval(Int2Type<RECIPROCAL>,
		Field<TM, IL, TL> const & f, TI s)
		DECL_RET_TYPE((1.0/f.get(s)) )

}
// namespace fetl_impl

template<typename TM, int IL, typename TL>
inline auto operator-(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<NEGATE,Field<TM,IL, TL> > > (f)))

template<typename TM, int IL, typename TL>
inline auto operator+(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (f))

template<typename TM, int IL, typename TL>
inline auto Reciprocal(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<RECIPROCAL,Field<TM,IL, TL> > > (f)))

//****************************************************************************************************
namespace fetl_impl
{

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<PLUS>,
		Field<TM, IL, TL> const &l, Field<TM, IL, TR> const &r, TI s)
		DECL_RET_TYPE((l.get(s)+r.get(s)))

}  // namespace fetl_impl

template<typename TM, int IL, typename TL, typename TR> inline auto //
operator+(Field<TM, IL, TL> const & lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE( ( Field<TM,IL , BiOp<PLUS,Field<TM,IL, TL> , Field<TM,IL, TR> > > (lhs, rhs)))

//****************************************************************************************************
namespace fetl_impl
{
template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<MINUS>,
		Field<TM, IL, TL> const &l, Field<TM, IL, TR> const &r, TI s)
		DECL_RET_TYPE((l.get(s)-r.get(s)))
}  // namespace fetl_impl
template<typename TM, int IL, typename TL, typename TR> inline auto //
operator-(Field<TM, IL, TL> const & lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE( ( Field<TM,IL , BiOp<MINUS,Field<TM,IL, TL> , Field<TM,IL, TR> > > (lhs, rhs)))

// *****************************************************************

namespace fetl_impl
{

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<MULTIPLIES>,
		Field<TM, IL, TL> const &l, TR const &r, TI s)
		ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l.get(s) * r))

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<MULTIPLIES>,
		TL const & l, Field<TM, IL, TR> const & r, TI s)
		ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l * r.get(s)))

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DIVIDES>,
		Field<TM, IL, TL> const &l, TR const &r, TI s)
		ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l.get(s) / r))

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DIVIDES>,
		TL const & l, Field<TM, IL, TR> const & r, TI s)
		ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l / r.get(s)))

} // namespace fetl_impl

template<typename TM, int IL, typename TL, int IR, typename TR> inline auto //
operator*(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE((Wedge(lhs,rhs)))
template<typename TM, typename TL, int IL, typename TR> inline auto //
operator*(TL lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,TL,Field<TM,IL ,TR> > > (lhs, rhs)))
template<typename TM, int IL, typename TL, typename TR> inline auto //
operator*(Field<TM, IL, TL> const & lhs, TR rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Field<TM,IL ,TL>,TR > > (lhs, rhs)))

template<typename TM, int IL, typename TL, int IR, typename TR> inline auto //
operator/(Field<TM, IL, TL> lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE(Wedge(lhs, Reciprocal(rhs)))

template<typename TM, int IL, typename TL, typename TR> inline auto //
operator/(TL lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,TL,Field<TM,IL ,TR> > > (lhs, rhs)))

template<typename TM, int IL, typename TL, typename TR> inline auto //
operator/(Field<TM, IL, TL> const & lhs, TR rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Field<TM,IL ,TL>,TR > > (lhs, rhs)))

//****************************************************************************************************

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************
template<typename TM, int IL, typename TL>
inline auto HodgeStar(Field<TM, IL, TL> const & f)
COND_DECL_RET_TYPE(
		(IL >= 0 && IL <= TM::NDIMS),

		( Field<TM, TM::NDIMS - IL , UniOp<HODGESTAR ,Field<TM,IL , TL> > >(f)),

		Zero )

template<typename TM, int IL, typename TL>
inline auto operator*(Field<TM, IL, TL> const & f)
DECL_RET_TYPE((HodgeStar(f)))
//****************************************************************************************************
template<typename TM, int IL, typename TL>
inline auto ExteriorDerivative(Field<TM, IL, TL> const & f)
COND_DECL_RET_TYPE(
		(IL >= 0 && IL < TM::NDIMS),

		( Field<TM, IL+1 ,UniOp<EXTRIORDERIVATIVE,Field<TM,IL , TL> > >(f)),

		Zero )

template<typename TM, int IL, typename TL>
inline auto d(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (ExteriorDerivative(f)) )

template<typename TM, int IL, typename TL, typename TR>
inline auto InteriorProduct(nTuple<TM::NDIMS, TR> const & v, Field<TM, IL, TR> const & f)
COND_DECL_RET_TYPE(
		(IL > 0 && IL <= TM::NDIMS),

		(Field<TM, IL+1 , BiOp<INTERIOR_PRODUCT, nTuple<TM::NDIMS, TR> ,Field<TM,IL , TL> > >(v,f)),

		Zero )

template<typename TM, int IL, typename TL, typename TR>
inline auto iv(nTuple<TM::NDIMS, TR> const & v, Field<TM, IL, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v,f)) )

template<typename TM, int IL, typename TL>
inline auto Codifferential(Field<TM, IL, TL> const & f)
COND_DECL_RET_TYPE(
		(IL > 0 && IL <= TM::NDIMS),

		(Field< TM, IL-1 , UniOp<CODIFFERENTIAL,Field<TM,IL , TL> > >( f)),

		Zero )

template<typename TM, int IL, int IR, typename TL, typename TR>
inline auto Wedge(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE( ( Field< TM,IL+IR , BiOp<WEDGE,Field<TM,IL,TL> , Field<TM,IR,TR> > > (lhs, rhs)))

template<typename TM, int IL, int IR, typename TL, typename TR>
inline auto operator^(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE( (Wedge(lhs,rhs)) )
//****************************************************************************************************
template<typename TG, int IL, typename TL, typename TR> inline auto //
InnerProduct(Field<TG, IL, TL> const & lhs, Field<TG, IL, TR> const & rhs)
DECL_RET_TYPE(Wedge (lhs,HodgeStar( rhs) ))

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<TG, EDGE, TL> const & lhs, Field<TG, FACE, TR> const & rhs)
DECL_RET_TYPE(Wedge(lhs , rhs ))

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<TG, FACE, TL> const & lhs, Field<TG, EDGE, TR> const & rhs)
DECL_RET_TYPE(Wedge(lhs , rhs ))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<TG, EDGE, TL> const & lhs, Field<TG, EDGE, TR> const & rhs)
DECL_RET_TYPE( Wedge(lhs , rhs ))

template<typename TG, typename TL, typename TR> inline auto //
Dot(nTuple<3, TL> const & v, Field<TG, EDGE, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v, f)))

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<TG, EDGE, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE( (InteriorProduct(v, f)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<TG, EDGE, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE( (InteriorProduct(v, HodgeStar(f))))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<TG, FACE, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE((InteriorProduct(v, f)))

template<typename TM, typename TR>
inline auto Grad(Field<TM, VERTEX, TR> const & f)
DECL_RET_TYPE(( ExteriorDerivative(f)))

template<typename TM, typename TR>
inline auto Diverge(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((Codifferential(-f)))

template<typename TM, typename TR>
inline auto Diverge(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative( f)))

template<typename TM, typename TR>
inline auto Curl(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f)))
template<typename TM, typename TR>
inline auto Curl(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f)))

////****************************************************************************************************
//
//namespace fetl_impl
//{
//template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DOT>,
//		Field<TM, VERTEX, TL> const &l, Field<TM, VERTEX, TR> const &r, TI s)
//		DECL_RET_TYPE((Dot(l.get(s) , r.get(s))))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DOT>,
//		Field<TM, VERTEX, TL> const &l, nTuple<3, TR> const &r, TI s)
//		DECL_RET_TYPE((Dot(l.get(s) , r)))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DOT>,
//		nTuple<3, TL> const & l, Field<TM, VERTEX, TR> const & r, TI s)
//		DECL_RET_TYPE((Dot(l , r.get(s))))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<CROSS>,
//		Field<TM, VERTEX, TL> const &l, Field<TM, VERTEX, TR> const &r, TI s)
//		DECL_RET_TYPE((Cross(l.get(s) , r.get(s))))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<CROSS>,
//		Field<TM, VERTEX, TL> const &l, nTuple<3, TR> const &r, TI s)
//		DECL_RET_TYPE((Cross(l.get(s) , r)))
//
//template<typename TM, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<CROSS>,
//		nTuple<3, TL> const & l, Field<TM, VERTEX, TR> const & r, TI s)
//		DECL_RET_TYPE((Cross(l , r.get(s))))
//
//} // namespace fetl_impl
//
//template<typename TG, typename TL, typename TR> inline auto //
//Dot(Field<TG, VERTEX, TL> const & lhs, Field<TG, VERTEX, TR> const & rhs)
//DECL_RET_TYPE( (Field<TG,VERTEX , BiOp<DOT,Field<TG,VERTEX, TL> ,
//				Field<TG,VERTEX, TR> > >(lhs, rhs)))
//
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
//template<typename TG, typename TL, typename TR> inline auto //
//Cross(Field<TG, VERTEX, TL> const & lhs, Field<TG, VERTEX, TR> const & rhs)
//DECL_RET_TYPE( (Field<TG,VERTEX , BiOp<CROSS,Field<TG,VERTEX, TL> ,
//				Field<TG,VERTEX, TR> > >(lhs, rhs)))
//
////******************************************************************************************************

namespace fetl_impl
{

// Check the availability of member function OpEval

HAS_MEMBER_FUNCTION(OpEval)

template<int TOP, typename TM, typename TL, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TI s)
ENABLE_IF_DECL_RET_TYPE(
		(has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TI >::value),
		(mesh.OpEval(Int2Type<TOP>(), l, s ))
)

template<int TOP, typename TM, typename TL, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TI s)
ENABLE_IF_DECL_RET_TYPE(
		(!has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TI>::value),
		(FieldOpEval(Int2Type<TOP>(), l, s))
)

template<int TOP, typename TM, typename TL, typename TR, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r, TI s)
ENABLE_IF_DECL_RET_TYPE(
		(has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TR const &,TI>::value),
		(mesh.OpEval(Int2Type<TOP>(), l,r, s))
)

template<int TOP, typename TM, typename TL, typename TR, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r, TI s)
ENABLE_IF_DECL_RET_TYPE(
		(!has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TR const &,TI>::value),
		(FieldOpEval(Int2Type<TOP>(), l,r, s))
)

}  // namespace fetl_impl

template<typename TM, int IFORM, int TOP, typename TL>
struct Field<TM, IFORM, UniOp<TOP, TL> >
{

public:

	typename StorageTraits<TL>::const_reference l_;

	typedef TM mesh_type;

	static constexpr int IForm = IFORM;

	typedef Field<TM, IForm, UniOp<TOP, TL> > this_type;

	typedef typename mesh_type::index_type index_type;

	mesh_type const & mesh;

	Field(TL const & l) :
			mesh(l.mesh), l_(l)
	{
	}

	inline auto get(index_type s) const
	DECL_RET_TYPE((fetl_impl::OpEval(Int2Type<TOP>(), mesh, l_, s)))

	inline auto operator[](index_type s) const
	DECL_RET_TYPE((get(s)))

};

template<typename TM, int IFORM, int TOP, typename TL, typename TR>
struct Field<TM, IFORM, BiOp<TOP, TL, TR> >
{

public:
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr int IForm = IFORM;

	typedef Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::index_type index_type;

	mesh_type const & mesh;

	Field(TL const & l, TR const & r) :
			mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}

	inline auto get(index_type s) const
	DECL_RET_TYPE((fetl_impl::OpEval(Int2Type<TOP>(), mesh, l_, r_, s)))

	inline auto operator[](index_type s) const
	DECL_RET_TYPE( (get(s)) )

private:

	template<int IL, typename VL, typename VR> static inline mesh_type const &
	get_mesh(Field<mesh_type, IL, VL> const & l, VR const & r)
	{
		return (l.mesh);
	}
	template<typename VL, int IR, typename VR> static inline mesh_type const &
	get_mesh(VL const & l, Field<mesh_type, IR, VR> const & r)
	{
		return (r.mesh);
	}

	template<int IL, typename VL, int IR, typename VR> static inline mesh_type const &
	get_mesh(Field<mesh_type, IL, VL> const & l, Field<mesh_type, IR, VR> const & r)
	{
		return (l.mesh);
	}

}
;

//****************************************************************************************************

}
// namespace simpla
#endif /* OPERATIONS_H_ */
