/*
 *  operations.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include <type_traits>

#include "field.h"

#include "../utilities/primitives.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/constant_ops.h"

namespace simpla
{

template<int, typename > class nTuple;

/**
 *  \defgroup  FETL Field expression template library
 *  @{
 *  	\defgroup  BasicAlgebra Basic algebra
 *  	\defgroup  ExteriorAlgebra Exterior algebra
 *  	\defgroup  VectorAlgebra Vector algebra
 *  	\defgroup  NonstandardOperations Non-standard operations
 *
 */



//! \ingroup   BasicAlgebra
//! @{
template<typename TM, int IL, typename TL, typename TR> inline auto operator==(Field<TM, IL, TL> const & lhs,
        Field<TM, IL, TR> const & rhs)
        DECL_RET_TYPE((lhs-rhs))
;

namespace fetl_impl
{

template<typename TM, int IL, typename TL, typename TI> inline auto FieldOpEval(Int2Type<NEGATE>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((-f.get(s)) )
;

template<typename TM, int IL, typename TL, typename TI> inline auto FieldOpEval(Int2Type<RECIPROCAL>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((1.0/f.get(s)) )
;

template<typename TM, int IL, typename TL, typename TI> inline auto FieldOpEval(Int2Type<REAL>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((real(f.get(s)) ))
;

template<typename TM, int IL, typename TL, typename TI> inline auto FieldOpEval(Int2Type<IMAGINE>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((imag(f.get(s)) ))
;

}

template<typename TM, int IL, typename TL>
inline auto operator-(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<NEGATE,Field<TM,IL, TL> > > (f)))
;

template<typename TM, int IL, typename TL>
inline auto Negate(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<NEGATE,Field<TM,IL, TL> > > (f)))
;

template<typename TM, int IL, typename TL>
inline auto operator+(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (f))
;

template<typename TM, int IL, typename TL>
inline auto Reciprocal(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<RECIPROCAL,Field<TM,IL, TL> > > (f)))
;

template<typename TM, int IR, typename TR>
inline auto real(Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IR ,UniOp<REAL, Field<TM,IR , TR> > >( f)))
;

template<typename TM, int IR, typename TR>
inline auto imag(Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IR ,UniOp<IMAGINE, Field<TM,IR , TR> > >( f)))
;

namespace fetl_impl
{

template<typename TM, int IL, typename TL, typename TR, typename TI>
inline auto FieldOpEval(Int2Type<PLUS>, Field<TM, IL, TL> const &l, Field<TM, IL, TR> const &r, TI s)
DECL_RET_TYPE((l.get(s)+r.get(s)))
;

template<typename TM, int IL, typename TL, typename TR, typename TI>
inline auto FieldOpEval(Int2Type<MINUS>, Field<TM, IL, TL> const &l, Field<TM, IL, TR> const &r, TI s)
DECL_RET_TYPE((l.get(s)-r.get(s)))
;

template<typename TM, typename TL, typename TI>
inline auto FieldOpEval(Int2Type<PLUS>, Field<TM, VERTEX, TL> const &l, Real r, TI s)
DECL_RET_TYPE((l.get(s)+r*l.mesh.Volume(s)) )
;

template<typename TM, typename TR, typename TI>
inline auto FieldOpEval(Int2Type<PLUS>, Real l, Field<TM, VERTEX, TR> const &r, TI s)
DECL_RET_TYPE((l*r.mesh.Volume(s) +r.get(s)))
;

template<typename TM, typename TL, typename TI>
inline auto FieldOpEval(Int2Type<MINUS>, Field<TM, VERTEX, TL> const &l, Real r, TI s)
DECL_RET_TYPE((l.get(s)-r*l.mesh.Volume(s)) )
;

template<typename TM, typename TR, typename TI>
inline auto FieldOpEval(Int2Type<MINUS>, Real l, Field<TM, VERTEX, TR> const &r, TI s)
DECL_RET_TYPE((l*r.mesh.Volume(s) -r.get(s)))
;

}

template<typename TM, int IL, typename TL, typename TR>
inline auto operator+(Field<TM, IL, TL> const & lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE( ( Field<TM,IL , BiOp<PLUS,Field<TM,IL, TL> , Field<TM,IL, TR> > > (lhs, rhs)))
;

template<typename TM, int IL, typename TL, typename TR>
inline auto operator-(Field<TM, IL, TL> const & lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE( ( Field<TM,IL , BiOp<MINUS,Field<TM,IL, TL> , Field<TM,IL, TR> > > (lhs, rhs)))
;

template<typename TM, typename TL>
inline auto operator+(Field<TM, VERTEX, TL> const & lhs, Real rhs)
DECL_RET_TYPE( ( Field<TM,VERTEX , BiOp<PLUS,Field<TM,VERTEX, TL> , Real > > (lhs, rhs)))
;

template<typename TM, typename TL>
inline auto operator-(Field<TM, VERTEX, TL> const & lhs, Real rhs)
DECL_RET_TYPE( ( Field<TM,VERTEX , BiOp<MINUS,Field<TM,VERTEX, TL> , Real > > (lhs, rhs)))
;

namespace fetl_impl
{

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<MULTIPLIES>,
        Field<TM, IL, TL> const &l, TR const &r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l.get(s) * r))
;

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<MULTIPLIES>,
        TL const & l, Field<TM, IL, TR> const & r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l * r.get(s)))
;

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DIVIDES>,
        Field<TM, IL, TL> const &l, TR const &r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l.get(s) / r))
;

template<typename TM, int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(Int2Type<DIVIDES>,
        TL const & l, Field<TM, IL, TR> const & r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l / r.get(s)))
;

}

template<typename TM, int IL, typename TL, int IR, typename TR> inline auto operator*(Field<TM, IL, TL> const & lhs,
        Field<TM, IR, TR> const & rhs)
        DECL_RET_TYPE((Wedge(lhs,rhs)))
;

template<typename TM, int IL, typename TL, int IR, typename TR> inline auto operator/(Field<TM, IL, TL> const & lhs,
        Field<TM, IR, TR> const & rhs)
        DECL_RET_TYPE((Wedge(lhs,Reciprocal(rhs))))
;

template<typename TM, int IL, typename TR> inline auto operator*(Real lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Real,Field<TM,IL ,TR> > > (lhs, rhs)))
;
template<typename TM, int IL, typename TL> inline auto operator*(Field<TM, IL, TL> const & lhs, Real rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Field<TM,IL ,TL>,Real > > (lhs, rhs)))
;

template<typename TM, int IL, typename TR> inline auto operator/(Real lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Real,Field<TM,IL ,TR> > > (lhs, rhs)))
;

template<typename TM, int IL, typename TL> inline auto operator/(Field<TM, IL, TL> const & lhs, Real rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Field<TM,IL ,TL>,Real > > (lhs, rhs)))
;

template<typename TM, int IL, typename TR> inline auto operator*(Complex lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Complex,Field<TM,IL ,TR> > > (lhs, rhs)))
;
template<typename TM, int IL, typename TL> inline auto operator*(Field<TM, IL, TL> const & lhs, Complex rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Field<TM,IL ,TL>,Complex > > (lhs, rhs)))
;

template<typename TM, int IL, typename TR> inline auto operator/(Complex lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Complex,Field<TM,IL ,TR> > > (lhs, rhs)))
;

template<typename TM, int IL, typename TL> inline auto operator/(Field<TM, IL, TL> const & lhs, Complex rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Field<TM,IL ,TL>,Complex > > (lhs, rhs)))
;
//! @}

//! \ingroup  ExteriorAlgebra
//! @{

template<typename TM, int IL, typename TL>
inline auto ExteriorDerivative(
        Field<TM, IL, TL> const & f)
                COND_DECL_RET_TYPE((IL >= 0 && IL < TM::NDIMS),( Field<TM, IL+1 ,UniOp<EXTRIORDERIVATIVE,Field<TM,IL , TL> > >(f)),Zero )
;

template<typename TM, int IL, typename TL>
inline auto HodgeStar(
        Field<TM, IL, TL> const & f)
                COND_DECL_RET_TYPE((IL <= TM::NDIMS && IL >= 0 ),( Field<TM, TM::NDIMS - IL , UniOp<HODGESTAR ,Field<TM,IL , TL> > >(f)),Zero )
;

template<typename TM, int IL, typename TL>
inline auto Codifferential(
        Field<TM, IL, TL> const & f)
                COND_DECL_RET_TYPE( (IL > 0 && IL <= TM::NDIMS),(Field< TM, IL-1 , UniOp<CODIFFERENTIAL,Field<TM,IL , TL> > >( f)), Zero )
;
template<typename TM, int IL, int IR, typename TL, typename TR>
inline auto Wedge(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE( ( Field< TM,IL+IR , BiOp<WEDGE,Field<TM,IL,TL> , Field<TM,IR,TR> > > (lhs, rhs)))
;

template<typename TM, int IL, typename TL, typename TR>
inline auto InteriorProduct(nTuple<TM::NDIMS, TR> const & v,
        Field<TM, IL, TR> const & f)
                COND_DECL_RET_TYPE( (IL > 0 && IL <= TM::NDIMS), (Field<TM, IL+1 , BiOp<INTERIOR_PRODUCT, nTuple<TM::NDIMS, TR> ,Field<TM,IL , TL> > >(v,f)),Zero )
;

template<typename TM, int IL, typename TL>
inline auto operator*(Field<TM, IL, TL> const & f)
DECL_RET_TYPE((HodgeStar(f)))
;
template<typename TM, int IL, typename TL>
inline auto d(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (ExteriorDerivative(f)) )
;

template<typename TM, int IL, typename TL>
inline auto delta(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (Codifferential(f)) )
;

template<typename TM, int IL, typename TL, typename TR>
inline auto iv(nTuple<TM::NDIMS, TR> const & v, Field<TM, IL, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v,f)) )
;

template<typename TM, int IL, int IR, typename TL, typename TR>
inline auto operator^(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE( (Wedge(lhs,rhs)) )
;

template<typename TM, int N, int IL, typename TL>
inline auto ExteriorDerivative(Field<TM, IL, TL> const & f,
        Int2Type<N>)
                COND_DECL_RET_TYPE((IL >= 0 && IL < TM::NDIMS), ( Field<TM, IL+1 ,BiOp<EXTRIORDERIVATIVE,Field<TM,IL , TL>,Int2Type<N>> >(f,Int2Type<N>())), Zero )
;

template<int N, typename TM, int IL, typename TL>
inline auto Codifferential(Field<TM, IL, TL> const & f,
        Int2Type<N>)
                COND_DECL_RET_TYPE((IL > 0 && IL <= TM::NDIMS), (Field< TM, IL-1 , BiOp<CODIFFERENTIAL,Field<TM,IL , TL> ,Int2Type<N>> >(f,Int2Type<N>() )), Zero )
;

//!  @}

//!  \ingroup  VectorAlgebra
//!  @{
template<typename TG, int IL, typename TL, typename TR> inline auto InnerProduct(Field<TG, IL, TL> const & lhs,
        Field<TG, IL, TR> const & rhs)
        DECL_RET_TYPE(Wedge (lhs,HodgeStar( rhs) ))
;

template<typename TG, typename TL, typename TR> inline auto Dot(Field<TG, EDGE, TL> const & lhs,
        Field<TG, FACE, TR> const & rhs)
        DECL_RET_TYPE(Wedge(lhs , rhs ))
;

template<typename TG, typename TL, typename TR> inline auto Dot(Field<TG, FACE, TL> const & lhs,
        Field<TG, EDGE, TR> const & rhs)
        DECL_RET_TYPE(Wedge(lhs , rhs ))
;

template<typename TG, typename TL, typename TR> inline auto Cross(Field<TG, EDGE, TL> const & lhs,
        Field<TG, EDGE, TR> const & rhs)
        DECL_RET_TYPE( Wedge(lhs , rhs ))
;

template<typename TG, typename TL, typename TR> inline auto Dot(nTuple<3, TL> const & v, Field<TG, EDGE, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v, f)))
;

template<typename TG, typename TL, typename TR> inline auto Dot(Field<TG, EDGE, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE( (InteriorProduct(v, f)))
;

template<typename TG, typename TL, typename TR> inline auto Cross(Field<TG, EDGE, TR> const & f,
        nTuple<3, TL> const & v)
        DECL_RET_TYPE( (InteriorProduct(v, HodgeStar(f))))
;

template<typename TG, typename TL, typename TR> inline auto Cross(Field<TG, FACE, TR> const & f,
        nTuple<3, TL> const & v)
        DECL_RET_TYPE((InteriorProduct(v, f)))
;

template<typename TM, typename TR>
inline auto Grad(Field<TM, VERTEX, TR> const & f)
DECL_RET_TYPE( ( ExteriorDerivative(f)))
;

template<typename TM, typename TR>
inline auto Diverge(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative( f)))
;

template<typename TM, typename TR>
inline auto Curl(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f)))
;

template<typename TM, typename TR>
inline auto Grad(Field<TM, VOLUME, TR> const & f)
DECL_RET_TYPE(Negate(Codifferential(f)))
;

template<typename TM, typename TR>
inline auto Diverge(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE(Negate(Codifferential(f)))
;

template<typename TM, typename TR>
inline auto Curl(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE(Negate(Codifferential(f)))
;

//!   @}

//!  \ingroup  NonstandardOperations
//!   @{
template<typename TM, typename TR>
inline auto CurlPDX(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f,Int2Type<0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f,Int2Type<1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f,Int2Type<2>())))
;

template<typename TM, typename TR>
inline auto CurlPDX(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f,Int2Type<0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f,Int2Type<1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f,Int2Type<2>())))
;

template<int IL, typename TM, int IR, typename TR>
inline auto MapTo(Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IL , BiOp<MAPTO,Int2Type<IL>,Field<TM,IR , TR> > >(Int2Type<IL>(), f)))
;

template<int IL, typename TM, int IR, typename TR>
inline auto MapTo(Int2Type<IL>, Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IL , BiOp<MAPTO,Int2Type<IL>,Field<TM,IR , TR> > >(Int2Type<IL>(), f)))
;

//!   @}

namespace fetl_impl
{

//! Check the availability of member function OpEval
HAS_MEMBER_FUNCTION(OpEval);

template<int TOP, typename TM, typename TL, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l,
        TI s)
                ENABLE_IF_DECL_RET_TYPE( (has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TI >::value), (mesh.OpEval(Int2Type<TOP>(), l, s )))
;

template<int TOP, typename TM, typename TL, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l,
        TI s)
                ENABLE_IF_DECL_RET_TYPE( (!has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TI>::value), (FieldOpEval(Int2Type<TOP>(), l, s)))
;

template<int TOP, typename TM, typename TL, typename TR, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r,
        TI s)
                ENABLE_IF_DECL_RET_TYPE( (has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TR const &,TI>::value), (mesh.OpEval(Int2Type<TOP>(), l,r, s)) )
;
template<int TOP, typename TM, typename TL, typename TR, typename TI>
auto OpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r,
        TI s)
                ENABLE_IF_DECL_RET_TYPE((!has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TR const &,TI>::value), (FieldOpEval(Int2Type<TOP>(), l,r, s)) )
;

}

/**
 *  \brief Bi-operation field expression
 */
template<typename TM, int IFORM, int TOP, typename TL, typename TR>
struct Field<TM, IFORM, BiOp<TOP, TL, TR> >
{

public:
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr int IForm = IFORM;

	typedef Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::iterator iterator;

	mesh_type const & mesh;

	Field(TL const & l, TR const & r)
			: mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}

	template<typename TI>
	inline auto get(TI s) const
	DECL_RET_TYPE((fetl_impl::OpEval(Int2Type<TOP>(), mesh, l_, r_, s)))
	;
	template<typename TI>
	inline auto operator[](TI s) const
	DECL_RET_TYPE( (this->get(s)) )
	;

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

/**
 *   \brief  Uni-operation field expression
 */
template<typename TM, int IFORM, int TOP, typename TL>
struct Field<TM, IFORM, UniOp<TOP, TL> >
{

public:

	typename StorageTraits<TL>::const_reference l_;

	typedef TM mesh_type;

	static constexpr int IForm = IFORM;

	typedef Field<TM, IForm, UniOp<TOP, TL> > this_type;

	typedef typename mesh_type::iterator iterator;

	mesh_type const & mesh;

	Field(TL const & l)
			: mesh(l.mesh), l_(l)
	{
	}
	template<typename TI>
	inline auto get(TI s) const
	DECL_RET_TYPE((fetl_impl::OpEval(Int2Type<TOP>(), mesh, l_, s)))
	;

	template<typename TI>
	inline auto operator[](TI s) const
	DECL_RET_TYPE((this->get(s)))
	;

};

template<typename TM, int IFORM, int TOP, typename TL, typename TR>
struct can_not_reference<Field<TM, IFORM, BiOp<TOP, TL, TR> >>
{
	static constexpr bool value = true;
};

template<typename TM, int IFORM, int TOP, typename TL>
struct can_not_reference<Field<TM, IFORM, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

//! @}

}//namespace simpla
#endif /* OPERATIONS_H_ */
