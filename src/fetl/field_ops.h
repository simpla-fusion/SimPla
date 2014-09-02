/*
 *  operations.h
 *
 *  created on: 2012-3-1
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

template<unsigned int, typename > class nTuple;

//! \ingroup FETL
//! \defgroup   BasicAlgebra Basic algebra
//! @{
template<typename TM, unsigned int IL, typename TL, typename TR> inline auto operator==(Field<TM, IL, TL> const & lhs,
        Field<TM, IL, TR> const & rhs)
        DECL_RET_TYPE((lhs-rhs))
;

namespace fetl_impl
{

template<typename TM, unsigned int IL, typename TL, typename TI> inline auto FieldOpEval(std::integral_constant<unsigned int ,NEGATE>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((-f.get(s)) )
;

template<typename TM, unsigned int IL, typename TL, typename TI> inline auto FieldOpEval(std::integral_constant<unsigned int ,RECIPROCAL>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((1.0/f.get(s)) )
;

template<typename TM, unsigned int IL, typename TL, typename TI> inline auto FieldOpEval(std::integral_constant<unsigned int ,REAL>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((real(f.get(s)) ))
;

template<typename TM, unsigned int IL, typename TL, typename TI> inline auto FieldOpEval(std::integral_constant<unsigned int ,IMAGINE>,
        Field<TM, IL, TL> const & f, TI s)
        DECL_RET_TYPE((imag(f.get(s)) ))
;

}

template<typename TM, unsigned int IL, typename TL>
inline auto operator-(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<NEGATE,Field<TM,IL, TL> > > (f)))
;

template<typename TM, unsigned int IL, typename TL>
inline auto Negate(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<NEGATE,Field<TM,IL, TL> > > (f)))
;

template<typename TM, unsigned int IL, typename TL>
inline auto operator+(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (f))
;

template<typename TM, unsigned int IL, typename TL>
inline auto Reciprocal(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( ( Field<TM,IL, UniOp<RECIPROCAL,Field<TM,IL, TL> > > (f)))
;

template<typename TM, unsigned int IR, typename TR>
inline auto real(Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IR ,UniOp<REAL, Field<TM,IR , TR> > >( f)))
;

template<typename TM, unsigned int IR, typename TR>
inline auto imag(Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IR ,UniOp<IMAGINE, Field<TM,IR , TR> > >( f)))
;

namespace fetl_impl
{

template<typename TM, unsigned int IL, typename TL, typename TR, typename TI>
inline auto FieldOpEval(std::integral_constant<unsigned int ,PLUS>, Field<TM, IL, TL> const &l, Field<TM, IL, TR> const &r, TI s)
DECL_RET_TYPE((l.get(s)+r.get(s)))
;

template<typename TM, unsigned int IL, typename TL, typename TR, typename TI>
inline auto FieldOpEval(std::integral_constant<unsigned int ,MINUS>, Field<TM, IL, TL> const &l, Field<TM, IL, TR> const &r, TI s)
DECL_RET_TYPE((l.get(s)-r.get(s)))
;

template<typename TM, typename TL, typename TI>
inline auto FieldOpEval(std::integral_constant<unsigned int ,PLUS>, Field<TM, VERTEX, TL> const &l, Real r, TI s)
DECL_RET_TYPE((l.get(s)+r*l.mesh.volume(s)) )
;

template<typename TM, typename TR, typename TI>
inline auto FieldOpEval(std::integral_constant<unsigned int ,PLUS>, Real l, Field<TM, VERTEX, TR> const &r, TI s)
DECL_RET_TYPE((l*r.mesh.volume(s) +r.get(s)))
;

template<typename TM, typename TL, typename TI>
inline auto FieldOpEval(std::integral_constant<unsigned int ,MINUS>, Field<TM, VERTEX, TL> const &l, Real r, TI s)
DECL_RET_TYPE((l.get(s)-r*l.mesh.volume(s)) )
;

template<typename TM, typename TR, typename TI>
inline auto FieldOpEval(std::integral_constant<unsigned int ,MINUS>, Real l, Field<TM, VERTEX, TR> const &r, TI s)
DECL_RET_TYPE((l*r.mesh.volume(s) -r.get(s)))
;

}

template<typename TM, unsigned int IL, typename TL, typename TR>
inline auto operator+(Field<TM, IL, TL> const & lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE( ( Field<TM,IL , BiOp<PLUS,Field<TM,IL, TL> , Field<TM,IL, TR> > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TL, typename TR>
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

template<typename TM, unsigned int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(
        std::integral_constant<unsigned int ,MULTIPLIES>, Field<TM, IL, TL> const &l, TR const &r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l.get(s) * r))
;

template<typename TM, unsigned int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(
        std::integral_constant<unsigned int ,MULTIPLIES>, TL const & l, Field<TM, IL, TR> const & r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l * r.get(s)))
;

template<typename TM, unsigned int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(std::integral_constant<unsigned int ,DIVIDES>,
        Field<TM, IL, TL> const &l, TR const &r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l.get(s) / r))
;

template<typename TM, unsigned int IL, typename TL, typename TR, typename TI> inline auto FieldOpEval(std::integral_constant<unsigned int ,DIVIDES>,
        TL const & l, Field<TM, IL, TR> const & r, TI s)
        ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value),(l / r.get(s)))
;

}

template<typename TM, unsigned int IL, typename TL, unsigned int IR, typename TR> inline auto operator*(
        Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
        DECL_RET_TYPE((Wedge(lhs,rhs)))
;

template<typename TM, unsigned int IL, typename TL, unsigned int IR, typename TR> inline auto operator/(
        Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
        DECL_RET_TYPE((Wedge(lhs,Reciprocal(rhs))))
;

template<typename TM, unsigned int IL, typename TR> inline auto operator*(Real lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Real,Field<TM,IL ,TR> > > (lhs, rhs)))
;
template<typename TM, unsigned int IL, typename TL> inline auto operator*(Field<TM, IL, TL> const & lhs, Real rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Field<TM,IL ,TL>,Real > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TR> inline auto operator/(Real lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Real,Field<TM,IL ,TR> > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TL> inline auto operator/(Field<TM, IL, TL> const & lhs, Real rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Field<TM,IL ,TL>,Real > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TR> inline auto operator*(Complex lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Complex,Field<TM,IL ,TR> > > (lhs, rhs)))
;
template<typename TM, unsigned int IL, typename TL> inline auto operator*(Field<TM, IL, TL> const & lhs, Complex rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<MULTIPLIES,Field<TM,IL ,TL>,Complex > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TR> inline auto operator/(Complex lhs, Field<TM, IL, TR> const & rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Complex,Field<TM,IL ,TR> > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TL> inline auto operator/(Field<TM, IL, TL> const & lhs, Complex rhs)
DECL_RET_TYPE((Field<TM,IL ,BiOp<DIVIDES,Field<TM,IL ,TL>,Complex > > (lhs, rhs)))
;
//! @}

//! \ingroup FETL
//! \defgroup  ExteriorAlgebra Exterior algebra
//! @{

template<typename TM, unsigned int IL, typename TL>
inline auto HodgeStar(Field<TM, IL, TL> const & f)
DECL_RET_TYPE((Field<TM, TM::NDIMS - IL, UniOp<HODGESTAR, Field<TM, IL, TL> > >(f)))

template<typename TM, unsigned int IL, typename TL>
inline auto ExteriorDerivative(Field<TM, IL, TL> const & f)
DECL_RET_TYPE(( Field<TM, IL+1 ,UniOp<EXTRIORDERIVATIVE,Field<TM,IL , TL> > >(f)) )
;

template<typename TM, typename TL>
inline Zero ExteriorDerivative(Field<TM, TM::NDIM, TL> const & f)
{
	return Zero();
}

template<typename TM, unsigned int IL, typename TL>
inline auto Codifferential(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (Field< TM, IL-1 , UniOp<CODIFFERENTIAL,Field<TM,IL , TL> > >( f)) )
;

template<typename TM, unsigned int IL, typename TL, typename ... Others>
inline Zero Codifferential(Field<TM, 0, TL> const & f, Others &&...)
{
	return Zero();
}

template<typename TM, unsigned int IL, unsigned int IR, typename TL, typename TR>
inline auto Wedge(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE( ( Field< TM,IL+IR , BiOp<WEDGE,Field<TM,IL,TL> , Field<TM,IR,TR> > > (lhs, rhs)))
;

template<typename TM, unsigned int IL, typename TL, typename TR>
inline Zero Wedge(Field<TM, IL, TL> const & lhs, Field<TM, TM::NDIMS - IL + 1, TR> const & rhs)
{
	return Zero();
}

template<typename TM, unsigned int IL, typename TL, typename TR>
inline auto InteriorProduct(nTuple<TM::NDIMS, TR> const & v, Field<TM, IL, TR> const & f)
DECL_RET_TYPE( (Field<TM, IL-1 , BiOp<INTERIOR_PRODUCT, nTuple<TM::NDIMS, TR> ,Field<TM,IL , TL> > >(v,f)))
;

template<typename TM, typename TL, typename TR>
inline Zero InteriorProduct(nTuple<TM::NDIMS, TR> const & v, Field<TM, 0, TR> const & f)
{
	return Zero();
}

template<typename TM, unsigned int IL, typename TL>
inline auto operator*(Field<TM, IL, TL> const & f)
DECL_RET_TYPE((HodgeStar(f)))
;
template<typename TM, unsigned int IL, typename TL>
inline auto d(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (ExteriorDerivative(f)) )
;

template<typename TM, unsigned int IL, typename TL>
inline auto delta(Field<TM, IL, TL> const & f)
DECL_RET_TYPE( (Codifferential(f)) )
;

template<typename TM, unsigned int IL, typename TL, typename TR>
inline auto iv(nTuple<TM::NDIMS, TR> const & v, Field<TM, IL, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v,f)) )
;

template<typename TM, unsigned int IL, unsigned int IR, typename TL, typename TR>
inline auto operator^(Field<TM, IL, TL> const & lhs, Field<TM, IR, TR> const & rhs)
DECL_RET_TYPE( (Wedge(lhs,rhs)) )
;

template<typename TM, unsigned int N, unsigned int IL, typename TL>
inline auto ExteriorDerivative(Field<TM, IL, TL> const & f, std::integral_constant<unsigned int ,N>)
DECL_RET_TYPE( ( Field<TM, IL+1 ,BiOp<EXTRIORDERIVATIVE,Field<TM,IL , TL>,std::integral_constant<unsigned int ,N>> >(f,std::integral_constant<unsigned int ,N>())))
;

template<unsigned int N, typename TM, unsigned int IL, typename TL>
inline auto Codifferential(Field<TM, IL, TL> const & f, std::integral_constant<unsigned int ,N>)
DECL_RET_TYPE( (Field< TM, IL-1 , BiOp<CODIFFERENTIAL,Field<TM,IL , TL> ,std::integral_constant<unsigned int ,N>> >(f,std::integral_constant<unsigned int ,N>() )) )
;

//!  @}

//!  \ingroup  FETL
//!  \defgroup  VectorAlgebra Vector algebra
//!  @{
template<typename TG, unsigned int IL, typename TL, typename TR> inline auto InnerProduct(Field<TG, IL, TL> const & lhs,
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

//!  \ingroup  FETL
//!  \defgroup  NonstandardOperations Non-standard operations
//!   @{
template<typename TM, typename TR>
inline auto CurlPDX(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(Field<TM, EDGE, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative(f,std::integral_constant<unsigned int ,2>())))
;

template<typename TM, typename TR>
inline auto CurlPDX(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(Field<TM, FACE, TR> const & f)
DECL_RET_TYPE((Codifferential(f,std::integral_constant<unsigned int ,2>())))
;

template<unsigned int IL, typename TM, unsigned int IR, typename TR>
inline auto MapTo(Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IL , BiOp<MAPTO,std::integral_constant<unsigned int ,IL>,Field<TM,IR , TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
;

template<unsigned int IL, typename TM, unsigned int IR, typename TR>
inline auto MapTo(std::integral_constant<unsigned int ,IL>, Field<TM, IR, TR> const & f)
DECL_RET_TYPE( (Field< TM, IL , BiOp<MAPTO,std::integral_constant<unsigned int ,IL>,Field<TM,IR , TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
;

//!   @}

namespace fetl_impl
{

//! Check the availability of member function OpEval
HAS_MEMBER_FUNCTION(OpEval);

template<unsigned int TOP, typename TM, typename TL, typename TI>
auto OpEval(std::integral_constant<unsigned int ,TOP>, TM const & mesh, TL const &l,
        TI s)
                ENABLE_IF_DECL_RET_TYPE( (has_member_function_OpEval<TM, std::integral_constant<unsigned int ,TOP>, TL const &,TI >::value), (mesh.OpEval(std::integral_constant<unsigned int ,TOP>(), l, s )))
;

template<unsigned int TOP, typename TM, typename TL, typename TI>
auto OpEval(std::integral_constant<unsigned int ,TOP>, TM const & mesh, TL const &l,
        TI s)
                ENABLE_IF_DECL_RET_TYPE( (!has_member_function_OpEval<TM, std::integral_constant<unsigned int ,TOP>, TL const &,TI>::value), (FieldOpEval(std::integral_constant<unsigned int ,TOP>(), l, s)))
;

template<unsigned int TOP, typename TM, typename TL, typename TR, typename TI>
auto OpEval(std::integral_constant<unsigned int ,TOP>, TM const & mesh, TL const &l, TR const &r,
        TI s)
                ENABLE_IF_DECL_RET_TYPE( (has_member_function_OpEval<TM, std::integral_constant<unsigned int ,TOP>, TL const &,TR const &,TI>::value), (mesh.OpEval(std::integral_constant<unsigned int ,TOP>(), l,r, s)) )
;
template<unsigned int TOP, typename TM, typename TL, typename TR, typename TI>
auto OpEval(std::integral_constant<unsigned int ,TOP>, TM const & mesh, TL const &l, TR const &r,
        TI s)
                ENABLE_IF_DECL_RET_TYPE((!has_member_function_OpEval<TM, std::integral_constant<unsigned int ,TOP>, TL const &,TR const &,TI>::value), (FieldOpEval(std::integral_constant<unsigned int ,TOP>(), l,r, s)) )
;

}

/**  \ingroup  FETL
 *  \brief Bi-operation field expression
 */
template<typename TM, unsigned int IFORM, unsigned int TOP, typename TL, typename TR>
struct Field<TM, IFORM, BiOp<TOP, TL, TR> >
{

public:
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;
	typedef TM mesh_type;

	static constexpr unsigned int IForm = IFORM;

	typedef Field<TM, IForm, BiOp<TOP, TL, TR> > this_type;

	typedef typename mesh_type::iterator iterator;

	mesh_type const & mesh;

	Field(TL const & l, TR const & r) :
			mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}

	template<typename TI>
	inline auto get(TI s) const
	DECL_RET_TYPE((fetl_impl::OpEval(std::integral_constant<unsigned int ,TOP>(), mesh, l_, r_, s)))
	;
	template<typename TI>
	inline auto operator[](TI s) const
	DECL_RET_TYPE( (get(s)) )
	;

private:

	template<unsigned int IL, typename VL, typename VR> static inline mesh_type const &
	get_mesh(Field<mesh_type, IL, VL> const & l, VR const & r)
	{
		return (l.mesh);
	}
	template<typename VL, unsigned int IR, typename VR> static inline mesh_type const &
	get_mesh(VL const & l, Field<mesh_type, IR, VR> const & r)
	{
		return (r.mesh);
	}

	template<unsigned int IL, typename VL, unsigned int IR, typename VR> static inline mesh_type const &
	get_mesh(Field<mesh_type, IL, VL> const & l, Field<mesh_type, IR, VR> const & r)
	{
		return (l.mesh);
	}

}
;

/**  \ingroup  FETL
 *   \brief  Uni-operation field expression
 */
template<typename TM, unsigned int IFORM, unsigned int TOP, typename TL>
struct Field<TM, IFORM, UniOp<TOP, TL> >
{

public:

	typename StorageTraits<TL>::const_reference l_;

	typedef TM mesh_type;

	static constexpr unsigned int IForm = IFORM;

	typedef Field<TM, IForm, UniOp<TOP, TL> > this_type;

	typedef typename mesh_type::iterator iterator;

	mesh_type const & mesh;

	Field(TL const & l) :
			mesh(l.mesh), l_(l)
	{
	}
	template<typename TI>
	inline auto get(TI s) const
	DECL_RET_TYPE((fetl_impl::OpEval(std::integral_constant<unsigned int ,TOP>(), mesh, l_, s)))
	;

	template<typename TI>
	inline auto operator[](TI s) const
	DECL_RET_TYPE((get(s)))
	;

};

template<typename TM, unsigned int IFORM, typename TExpr, typename TI>
auto get_value(Field<TM, IFORM, TExpr> const & f, TI const & s)->decltype(f.get(s))
{
	return f.get(s);
}

template<typename TM, unsigned int IFORM, unsigned int TOP, typename TL, typename TR>
struct can_not_reference<Field<TM, IFORM, BiOp<TOP, TL, TR> >>
{
	static constexpr bool value = true;
};

template<typename TM, unsigned int IFORM, unsigned int TOP, typename TL>
struct can_not_reference<Field<TM, IFORM, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

//! @}

}//namespace simpla
#endif /* OPERATIONS_H_ */
