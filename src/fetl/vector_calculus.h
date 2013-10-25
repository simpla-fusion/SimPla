/*
 *  _fetl_impl::vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_

#include "field.h"
#include "geometry.h"
#include "expression.h"
namespace simpla
{

template<typename, int> struct Geometry;

template<typename TGeometry, template<typename > class TOP, typename TL>
struct Field<TGeometry, UniOp<TOP, TL> > : public TGeometry
{
	typename ConstReferenceTraits<TL>::type l_;

	Field(TL const & l) :
			TGeometry(l), l_(l)
	{
	}
	inline auto operator[](size_t s) const
	DECL_RET_TYPE((TOP<TL>::eval(*TGeometry::mesh,l_ ,s)))

};

#define DEF_OP_CLASS(_NAME_)                                                  \
template<typename> struct OpF##_NAME_;                                        \
template<typename TG,typename TL>                                     \
struct OpF##_NAME_< Field<TG,TL> >                    \
{    \
	static inline auto eval(typename TG::Mesh const & mesh,  Field<TG,TL> const & l,size_t s)                                      \
	DECL_RET_TYPE((mesh._NAME_(l, s)))                         \
};                                                                            \

DEF_OP_CLASS(Grad)
DEF_OP_CLASS(Diverge)
DEF_OP_CLASS(Curl)
DEF_OP_CLASS(HodgeStar)
DEF_OP_CLASS(ExtriorDerivative)
DEF_OP_CLASS(Negate)
#undef DEF_OP_CLASS

template<typename TG, typename TR> inline auto //
Grad(Field<Geometry<TG, 0>, TR> const & f)
DECL_RET_TYPE ((Field<Geometry<TG, 1>,
				UniOp<OpFGrad,Field<Geometry<TG, 0>, TR> > >(f)))

template<typename TG, typename TR> inline auto //
Diverge(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE((Field<Geometry<TG, 0>,
				UniOp<OpFDiverge, Field<Geometry<TG, 1>, TR> > >( f)))

template<typename TG, typename TR> inline auto //
Curl(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				UniOp<OpFCurl, Field<Geometry<TG, 1>, TR> > >(f)))
template<typename TG, typename TR> inline auto //
Curl(Field<Geometry<TG, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				UniOp<OpFCurl, Field<Geometry<TG, 2>, TR> > >(f)))

//template<int> struct OpCurlPD;
//template<typename TGeometry, int IPD, typename TL>
//struct Field<TGeometry, UniOp<OpCurlPD<IPD>, TL> > : public TGeometry
//{
//	typename ConstReferenceTraits<TL>::type expr;
//	Field(TL const & l) :
//			TGeometry(l), expr(l)
//	{
//	}
//	inline auto operator[](size_t s) const
//	DECL_RET_TYPE((TGeometry::mesh.CurlPD(Int2Type<IPD>(),expr, s)))
//
//};
//#define DECL_RET_TYPE_1_ARG(_TYPE_,_ARG_) ->_TYPE_{return (_TYPE_(_ARG_);}
//
//template<int IPD, typename TG, typename TR> inline auto //
//CurlPD(Int2Type<2>, Field<Geometry<TG, 2>, TR> const & f)
//->Field<Geometry<TG, 1>, UniOp<OpCurlPD<IPD>,Field<Geometry<TG, 2>,TR > > >
//{
//	return (Field<Geometry<TG, 1>,
//			UniOp<OpCurlPD<IPD>, Field<Geometry<TG, 2>, TR> > >(f));
//}

template<typename TG, int IL, typename TL> inline  //
auto operator*(
		Field<Geometry<TG, IL>, TL> const & f)
				DECL_RET_TYPE(
						(typename std::conditional<(IL > 0 && IL <= TG::NUM_OF_DIMS),
								Field<Geometry<TG, TG::NUM_OF_DIMS - IL>,
								UniOp<OpFHodgeStar
								,Field<Geometry<TG, IL>, TL> > >, Zero>::type(f)))

template<typename TG, int IL, typename TL> inline  //
auto d(Field<Geometry<TG, IL>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IL > 0 && IL+1 <= TG::NUM_OF_DIMS),
				Field<Geometry<TG, IL+1>,
				UniOp<OpFExtriorDerivative,Field<Geometry<TG, IL>, TL> > >
				, Zero>::type(f)) )

template<typename TG, int IL, typename TL> inline  //
auto operator-(Field<Geometry<TG, IL>, TL> const & f)
DECL_RET_TYPE(
		( Field<Geometry<TG, IL>,
				UniOp<OpFNegate,Field<Geometry<TG, IL>, TL> > > (f)))

#define DEF_BIOP_CLASS(_NAME_)                                                  \
template<typename, typename > struct OpF##_NAME_;                                 \
template<typename TG,int IL, typename TL,int IR, typename TR>                                  \
struct OpF##_NAME_<Field<Geometry<TG, IL>, TL>, Field<Geometry<TG, IR>, TR> >                                \
{                                                                                \
	static inline auto eval(TG const &mesh,  Field<Geometry<TG, IL>, TL> const & l,            \
			Field<Geometry<TG, IR>, TR> const &r, size_t s)                                    \
					DECL_RET_TYPE ((mesh._NAME_(l, r, s)))                      \
};                                                                               \
template<typename TG,int IL, typename TL, typename TR>                                  \
struct OpF##_NAME_<Field<Geometry<TG,IL>, TL>, TR>                                             \
{                                                                                \
	static inline auto eval(TG const &mesh,  Field<Geometry<TG,IL>, TL> const & l,            \
			TR const &r, size_t s) DECL_RET_TYPE ((mesh._NAME_(l, r, s)))       \
};                                                                               \
template<typename TG, typename TL, int IR, typename TR>                                  \
struct OpF##_NAME_<TL, Field<Geometry<TG,IR>, TR>>                                             \
{                                                                                \
	static inline auto eval( TG const &mesh,  TL const & l, \
			Field<Geometry<TG,IR>, TR> const &r, size_t s) \
		DECL_RET_TYPE ((mesh._NAME_(l, r, s))) \
};

DEF_BIOP_CLASS(Wedge)
DEF_BIOP_CLASS(Plus)
DEF_BIOP_CLASS(Minus)
DEF_BIOP_CLASS(Multiplies)
DEF_BIOP_CLASS(Divides)
DEF_BIOP_CLASS(Cross)
DEF_BIOP_CLASS(Dot)
#undef DEF_BIOP_CLASS

template<typename TGeometry, template<typename, typename > class TOP,
		typename TL, typename TR>
struct Field<TGeometry, BiOp<TOP, TL, TR> > : public TGeometry
{
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;

	Field(TL const & l, TR const & r) :
			TGeometry(l, r), l_(l), r_(r)
	{
	}
	inline auto operator[](size_t s) const
	DECL_RET_TYPE((TOP<TL,TR>::eval(*TGeometry::mesh,l_,r_,s)))

};

template<typename TG, int IL, int IR, typename TL, typename TR> inline auto //
operator^(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, IR>, TR> const & rhs)
				DECL_RET_TYPE(
						(typename std::conditional<(IL + IR >=0 &&
										IL+IR <= TG::NUM_OF_DIMS ),
								Field<Geometry<TG,IL+IR> ,
								BiOp<OpFWedge,Field<Geometry<TG, IL>, TL> ,
								Field<Geometry<TG, IR>, TR> > >
								,Zero>::type
								(lhs, rhs)))

template<typename TGeo, typename TL, typename TR> inline auto //
operator+(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE(
		( Field<TGeo ,
				BiOp<OpFPlus,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))

template<typename TGeo, typename TL, typename TR> inline auto //
operator-(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE(
		( Field<TGeo , BiOp<OpFMinus,
				Field<TGeo, TL> ,
				Field<TGeo, TR> > > (lhs, rhs)))

template<typename TG, int IL, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,IL >,
						BiOp<OpFMultiplies,Field<Geometry<TG, IL>, TL>,
						Field<Geometry<TG,0>, TR> > > (lhs, rhs))
		)

template<typename TG, int IR, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TG, 0>, TL> const & lhs,
		Field<Geometry<TG, IR>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,IR >,
						BiOp<OpFMultiplies,Field<Geometry<TG, 0>, TL>,
						Field<Geometry<TG,IR>, TR> > > (lhs, rhs))
		)

template<typename TG, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TG, 0>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,0 >,
						BiOp<OpFMultiplies,Field<Geometry<TG, 0>, TL>,
						Field<Geometry<TG,0>, TR> > > (lhs, rhs))
		)

template<typename TG, typename TL, typename TR> inline auto //
operator*(Field<TG, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE(
		(Field<TG,BiOp<OpFMultiplies,Field<TG,TL>,TR > > (lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
operator*(TL const & lhs, Field<TG, TR> const & rhs)
DECL_RET_TYPE((Field<TG,BiOp<OpFMultiplies,TL,Field<TG,TR> > > (lhs, rhs)))

template<typename TG, int IL, typename TL, typename TR> inline auto //
operator/(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,IL >,
						BiOp<OpFDivides,Field<Geometry<TG, IL>, TL>,
						Field<Geometry<TG, 0>, TR> > > (lhs, rhs))
		)
template<typename TG, typename TL, typename TR> inline auto //
operator/(Field<Geometry<TG, 0>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,0 >,
						BiOp<OpFDivides,
						Field<Geometry<TG, 0>, TL>,
						Field<Geometry<TG, 0>, TR> > > (lhs, rhs))
		)
template<typename TG, int IL, typename TL, typename TR> inline auto //
operator/(Field<Geometry<TG, IL>, TL> const & lhs, TR rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IL >,
				BiOp<OpFDivides,Field<Geometry<TG, IL>, TL>, TR > > (lhs, rhs))
)

template<typename TGeo, typename TL, typename TR> inline auto //
operator==(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((lhs-rhs))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, 0>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
				DECL_RET_TYPE(
						(Field<Geometry<TG,0> ,
								BiOp<OpFCross,Field<Geometry<TG, 0>, TL> ,
								Field<Geometry<TG, 0>, TR> > >(lhs, rhs)))
template<typename TG, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, 0>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
				DECL_RET_TYPE(
						(Field<Geometry<TG,0> ,
								BiOp<OpFDot,Field<Geometry<TG, 0>, TL> ,
								Field<Geometry<TG, 0>, TR> > >(lhs, rhs)))

}
// namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
