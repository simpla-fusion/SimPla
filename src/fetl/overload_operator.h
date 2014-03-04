/*
 * overload_operator.h
 *
 *  Created on: 2013年12月17日
 *      Author: salmon
 */

#ifndef OVERLOAD_OPERATOR_H_
#define OVERLOAD_OPERATOR_H_

#include "ntuple.h"
#include "ntuple_ops.h"
#include "field.h"
#include "field_ops.h"
#include "constant_ops.h"
namespace simpla
{
//****************************************************************************************************
//********************       nTuple     Operation                      ****************
//****************************************************************************************************

// Expression template of nTuple

#define _DEFINE_BINARY_OPERATOR(_NAME_,_OP_)                                                \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)                   \
DECL_RET_TYPE((nTuple<N, BiOp<_NAME_ ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)))             \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (nTuple<N, TL> const & lhs, TR const & rhs)                              \
DECL_RET_TYPE((nTuple<N, BiOp<_NAME_, nTuple<N, TL>, TR> >(lhs, rhs)))                   \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (TL const & lhs, nTuple<N, TR> const & rhs)                              \
DECL_RET_TYPE((nTuple<N, BiOp<_NAME_, TL, nTuple<N, TR> > >(lhs, rhs)))              \


//_DEFINE_BINARY_OPERATOR(PLUS, +)
//_DEFINE_BINARY_OPERATOR(MINUS, -)
//_DEFINE_BINARY_OPERATOR(MULTIPLIES, *)
//_DEFINE_BINARY_OPERATOR(DIVIDES, /)
//_DEFINE_BINARY_OPERATOR(BITWISEXOR, ^)
//_DEFINE_BINARY_OPERATOR(BITWISEAND, &)
//_DEFINE_BINARY_OPERATOR(BITWISEOR, |)
//_DEFINE_BINARY_OPERATOR(MODULUS, %)

#undef _DEFINE_BINARY_OPERATOR

template<int N, typename TL, typename TR> inline auto operator +(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
->decltype(((nTuple<N, BiOp<PLUS ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))
{
	return ((nTuple<N, BiOp<PLUS, nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)));
}

//template<int N, typename TL, typename TR> inline auto operator +(nTuple<N, TL> const & lhs, TR const & rhs)
//->decltype(((nTuple<N, BiOp<PLUS, nTuple<N, TL>, TR> >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<PLUS, nTuple<N, TL>, TR> >(lhs, rhs)));
//}
//
//template<int N, typename TL, typename TR> inline auto operator +(TL const & lhs, nTuple<N, TR> const & rhs)
//->decltype(((nTuple<N, BiOp<PLUS, TL, nTuple<N, TR> > >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<PLUS, TL, nTuple<N, TR> > >(lhs, rhs)));
//}

template<int N, typename TL, typename TR> inline auto operator -(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
->decltype(((nTuple<N, BiOp<MINUS ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))
{
	return ((nTuple<N, BiOp<MINUS, nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)));
}

//template<int N, typename TL, typename TR> inline auto operator -(nTuple<N, TL> const & lhs, TR const & rhs)
//->decltype(((nTuple<N, BiOp<MINUS, nTuple<N, TL>, TR> >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<MINUS, nTuple<N, TL>, TR> >(lhs, rhs)));
//}
//
//template<int N, typename TL, typename TR> inline auto operator -(TL const & lhs, nTuple<N, TR> const & rhs)
//->decltype(((nTuple<N, BiOp<MINUS, TL, nTuple<N, TR> > >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<MINUS, TL, nTuple<N, TR> > >(lhs, rhs)));
//}

//template<int N, typename TL, typename TR> inline auto operator *(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
//->decltype(((nTuple<N, BiOp<MULTIPLIES ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<MULTIPLIES, nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)));
//}

template<int N, typename TL> inline auto operator *(nTuple<N, TL> const & lhs, double const & rhs)
->decltype(((nTuple<N, BiOp<MULTIPLIES, nTuple<N, TL>, double> >(lhs, rhs))))
{
	return ((nTuple<N, BiOp<MULTIPLIES, nTuple<N, TL>, double> >(lhs, rhs)));
}

template<int N, typename TR> inline auto operator *(double const & lhs, nTuple<N, TR> const & rhs)
->decltype(((nTuple<N, BiOp<MULTIPLIES, double, nTuple<N, TR> > >(lhs, rhs))))
{
	return ((nTuple<N, BiOp<MULTIPLIES, double, nTuple<N, TR> > >(lhs, rhs)));
}
//template<int N, typename TL, typename TR> inline auto operator /(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
//->decltype(((nTuple<N, BiOp<DIVIDES ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<DIVIDES, nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)));
//}

template<int N, typename TL, typename TR> inline auto operator /(nTuple<N, TL> const & lhs, TR const & rhs)
->decltype(((nTuple<N, BiOp<DIVIDES, nTuple<N, TL>, TR> >(lhs, rhs))))
{
	return ((nTuple<N, BiOp<DIVIDES, nTuple<N, TL>, TR> >(lhs, rhs)));
}

//template<int N, typename TL, typename TR> inline auto operator /(TL const & lhs, nTuple<N, TR> const & rhs)
//->decltype(((nTuple<N, BiOp<DIVIDES, TL, nTuple<N, TR> > >(lhs, rhs))))
//{
//	return ((nTuple<N, BiOp<DIVIDES, TL, nTuple<N, TR> > >(lhs, rhs)));
//}

template<int N, typename TL> inline
auto operator-(nTuple<N, TL> const & f)
DECL_RET_TYPE(( nTuple<N, UniOp<NEGATE,nTuple<N, TL> > > (f)))

template<int N, typename TL> inline
auto operator+(nTuple<N, TL> const & f)
DECL_RET_TYPE(f)

//template<int N, typename TL> inline auto operator -(Zero const &, nTuple<N, TL> const &f)
//DECL_RET_TYPE (( nTuple<N, UniOp<NEGATE,nTuple<N, TL> > > (f)))

template<int N, typename TL, typename TR> inline auto Cross(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE((nTuple<N,BiOp<CROSS, nTuple<N, TL>,nTuple<N, TR> > > (lhs, rhs)))

template<int N, typename TL, typename TR>
inline auto Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((ntuple_impl::_inner_product(l,r)))

//****************************************************************************************************
//*************************      Field Operation      ********************************************
//****************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto operator-(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(( Field<Geometry<TM, IFORM>, UniOp<NEGATE,Field<Geometry<TM, IFORM>, TL> > > (f)))
//****************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto operator+(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE((f))
//****************************************************************************************************
template<typename TGeo, typename TL, typename TR> inline auto //
operator-(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((Field<TGeo , BiOp<MINUS,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))
//****************************************************************************************************
template<typename TGeo, typename TL, typename TR> inline auto //
operator+(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((Field<TGeo , BiOp<PLUS,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))
//****************************************************************************************************

template<typename TM, int IL, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TM, IL>, TL> const & lhs, Field<Geometry<TM, 0>, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TM,IL >,BiOp<MULTIPLIES,
				Field<Geometry<TM,IL>,TL>, Field<Geometry<TM,0>,TR> > > (lhs, rhs)))

template<typename TM, int IR, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TM, 0>, TL> const & lhs, Field<Geometry<TM, IR>, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TM,IR >,BiOp<MULTIPLIES,
				Field<Geometry<TM,0>,TL>, Field<Geometry<TM,IR>,TR> > > (lhs, rhs)))

template<typename TM, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TM, 0>, TL> const & lhs, Field<Geometry<TM, 0>, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TM,0>,BiOp<MULTIPLIES,
				Field<Geometry<TM,0>,TL>, Field<Geometry<TM,0>,TR> > > (lhs, rhs)))

//template<typename TGeo, typename TL, typename TR> inline auto //
//operator*(Field<TGeo, TL> const & lhs, TR const & rhs)
//DECL_RET_TYPE((Field<TGeo,BiOp<MULTIPLIES,Field<TGeo,TL>,TR > > (lhs, rhs)))
//
//template<typename TGeo, typename TL, typename TR> inline auto //
//operator*(TL const & lhs, Field<TGeo, TR> const & rhs)
//DECL_RET_TYPE((Field<TGeo,BiOp<MULTIPLIES,TL,Field<TGeo,TR> > > (lhs, rhs)))

//// To remve the ambiguity of operator define
//template<typename TG, typename TL, int NR, typename TR> inline auto //
//operator*(Field<TG, TL> const & lhs, nTuple<NR, TR> const & rhs)
//DECL_RET_TYPE((Field<TG, BiOp<MULTIPLIES,Field<TG,TL>,nTuple<NR, TR> > > (lhs, rhs)))
//
//template<typename TG, int NL, typename TL, typename TR> inline auto //
//operator*(nTuple<NL, TL> const & lhs, Field<TG, TR> const & rhs)
//DECL_RET_TYPE((Field<TG, BiOp<MULTIPLIES,nTuple<NL,TL>,Field<TG,TR> > > (lhs, rhs)))

// *****************************************************************
template<typename TM, int IFORM, typename TL, typename TR> inline auto //
operator/(Field<Geometry<TM, IFORM>, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE( (Field<Geometry<TM,IFORM >,
				BiOp<DIVIDES,Field<Geometry<TM, IFORM>, TL>,TR > > (lhs, rhs)))

template<typename TG, int IFORM, typename TL, int NR, typename TR> inline auto //
operator/(Field<Geometry<TG, IFORM>, TL> const & lhs, nTuple<NR, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TG,IFORM >,
				BiOp<DIVIDES,Field<Geometry<TG, IFORM>, TL>, nTuple<NR,TR> > > (lhs, rhs)))

//****************************************************************************************************
template<typename TGeo, typename TL, typename TR> inline auto //
operator==(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((lhs-rhs))
//****************************************************************************************************

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto HodgeStar(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IFORM >= 0 && IFORM <= TM::NUM_OF_DIMS),
				Field<Geometry<TM, TM::NUM_OF_DIMS - IFORM>,
				UniOp<HODGESTAR ,Field<Geometry<TM, IFORM>, TL> > >, Zero>::type(f)))

template<typename TM, int IFORM, typename TL>
inline auto operator*(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE((HodgeStar(f)))
//****************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto ExteriorDerivative(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IFORM >= 0 && IFORM < TM::NUM_OF_DIMS),
				Field<Geometry<TM, IFORM+1>,
				UniOp<EXTRIORDERIVATIVE,Field<Geometry<TM, IFORM>, TL> > > , Zero>::type(f)) )

template<typename TM, int IFORM, typename TL>
inline auto d(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE( (ExteriorDerivative(f)) )
//****************************************************************************************************
template<typename TM, int IFORM, typename TL, typename TR>
inline auto InteriorProduct(nTuple<TM::NDIMS, TR> const & v, Field<Geometry<TM, IFORM>, TR> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IFORM > 0 && IFORM <= TM::NUM_OF_DIMS),
				Field<Geometry<TM, IFORM+1>,
				BiOp<INTERIOR_PRODUCT, nTuple<TM::NDIMS, TR> ,Field<Geometry<TM, IFORM>, TL> > > , Zero>::type( v,f)) )

template<typename TM, int IFORM, typename TL, typename TR>
inline auto i(nTuple<TM::NDIMS, TR> const & v, Field<Geometry<TM, IFORM>, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v,f)) )

//****************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto Codifferential(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IFORM > 0 && IFORM <= TM::NUM_OF_DIMS),
				Field<Geometry<TM, IFORM+1>,
				UniOp<CODIFFERENTIAL,Field<Geometry<TM, IFORM>, TL> > > , Zero>::type(f)) )

//****************************************************************************************************
template<typename TM, int IFORM, int IR, typename TL, typename TR>
inline auto Wedge(Field<Geometry<TM, IFORM>, TL> const & lhs, Field<Geometry<TM, IR>, TR> const & rhs)
DECL_RET_TYPE( ( Field<Geometry<TM,IFORM+IR> ,
				BiOp<WEDGE,Field<Geometry<TM, IFORM>, TL> , Field<Geometry<TM, IR>, TR> > > (lhs, rhs)))

template<typename TM, int IFORM, int IR, typename TL, typename TR>
inline auto operator^(Field<Geometry<TM, IFORM>, TL> const & lhs, Field<Geometry<TM, IR>, TR> const & rhs)
DECL_RET_TYPE( (Wedge(lhs,rhs)) )
//****************************************************************************************************


template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, 0>, TL> const & lhs, Field<Geometry<TG, 0>, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TG,0> , BiOp<CROSS,Field<Geometry<TG, 0>, TL> ,
				Field<Geometry<TG, 0>, TR> > >(lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, 0>, TL> const & lhs, nTuple<3, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TG,0> , BiOp<CROSS,Field<Geometry<TG, 0>, TL> ,
				nTuple<3,TR> > >(lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(nTuple<3, TL> const & lhs, Field<Geometry<TG, 0>, TR> const & rhs)
DECL_RET_TYPE( (Field<Geometry<TG,0> , BiOp<CROSS,nTuple<3,TL> ,
				Field<Geometry<TG, 0>, TR> > >(lhs, rhs)))

template<typename TG, int IFORM, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, IFORM>, TL> const & lhs, Field<Geometry<TG, IFORM>, TR> const & rhs)
DECL_RET_TYPE( (lhs ^(*rhs) ))

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, EDGE>, TL> const & lhs, Field<Geometry<TG, FACE>, TR> const & rhs)
DECL_RET_TYPE( (lhs ^ rhs ))

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, FACE>, TL> const & lhs, Field<Geometry<TG, EDGE>, TR> const & rhs)
DECL_RET_TYPE( (lhs ^ rhs ))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, EDGE>, TL> const & lhs, Field<Geometry<TG, EDGE>, TR> const & rhs)
DECL_RET_TYPE( (lhs ^ rhs ))

template<typename TG, typename TL, typename TR> inline auto //
Dot(nTuple<3, TL> const & v, Field<Geometry<TG, EDGE>, TR> const & f)
DECL_RET_TYPE( (InteriorProduct(v, f)))

template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, EDGE>, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE( (InteriorProduct(v, f)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, EDGE>, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE( (InteriorProduct(v, HodgeStar(f))))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, FACE>, TR> const & f, nTuple<3, TL> const & v)
DECL_RET_TYPE( (InteriorProduct(v, f)))

//****************************************************************************************************
template<typename TM, typename TR>
inline auto Grad(Field<Geometry<TM, VERTEX>, TR> const & f)
DECL_RET_TYPE(( ExteriorDerivative(f)))
//****************************************************************************************************
template<typename TM, typename TR>
inline auto Diverge(Field<Geometry<TM, EDGE>, TR> const & f)
DECL_RET_TYPE((-Codifferential( f)))

template<typename TM, typename TR>
inline auto Diverge(Field<Geometry<TM, FACE>, TR> const & f)
DECL_RET_TYPE((ExteriorDerivative( f)))

//****************************************************************************************************

template<typename TM, typename TR>
inline auto Curl(Field<Geometry<TM, EDGE>, TR> const & f)
DECL_RET_TYPE( (ExteriorDerivative(f)))
template<typename TM, typename TR>
inline auto Curl(Field<Geometry<TM, FACE>, TR> const & f)
DECL_RET_TYPE( (Codifferential(f)))

//template<typename TM, typename TR>
//inline auto CurlPDX(Field<Geometry<TM, 1>, TR> const & f)
//DECL_RET_TYPE( (Field<Geometry<TM, 2>, UniOp<CURLPDX, Field<Geometry<TM, 1>, TR> > >(f)))
//template<typename TM, typename TR>
//inline auto CurlPDY(Field<Geometry<TM, 1>, TR> const & f)
//DECL_RET_TYPE( (Field<Geometry<TM, 2>, UniOp<CURLPDY, Field<Geometry<TM, 1>, TR> > >(f)))
//template<typename TM, typename TR>
//inline auto CurlPDZ(Field<Geometry<TM, 1>, TR> const & f)
//DECL_RET_TYPE( (Field<Geometry<TM, 2>, UniOp<CURLPDZ, Field<Geometry<TM, 1>, TR> > >(f)))
//
//template<typename TM, typename TR>
//inline auto CurlPDX(Field<Geometry<TM, 2>, TR> const & f)
//DECL_RET_TYPE( (Field<Geometry<TM, 1>, UniOp<CURLPDX, Field<Geometry<TM, 2>, TR> > >(f)))
//template<typename TM, typename TR>
//inline auto CurlPDY(Field<Geometry<TM, 2>, TR> const & f)
//DECL_RET_TYPE( (Field<Geometry<TM, 1>, UniOp<CURLPDY, Field<Geometry<TM, 2>, TR> > >(f)))
//template<typename TM, typename TR>
//inline auto CurlPDZ(Field<Geometry<TM, 2>, TR> const & f)
//DECL_RET_TYPE( (Field<Geometry<TM, 1>, UniOp<CURLPDZ, Field<Geometry<TM, 2>, TR> > >(f)))

}// namespace simpla

#endif /* OVERLOAD_OPERATOR_H_ */
