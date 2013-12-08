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
#include "ntuple.h"
#include "primitives.h"

namespace simpla
{

namespace fetl_impl
{

template<int N, typename TL, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<NEGATE>, Field<Geometry<TM, N>, TL> const & f, TI ... s)
		DECL_RET_TYPE((-f.get(s...)) )

template<int IL, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<PLUS>, Field<Geometry<TM, IL>, TL> const &l,
		Field<Geometry<TM, IL>, TR> const &r, TI ... s)
		DECL_RET_TYPE((l.get(s...)+r.get(s...)))

template<int IL, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<MINUS>, Field<Geometry<TM, IL>, TL> const &l,
		Field<Geometry<TM, IL>, TR> const &r, TI ... s)
		DECL_RET_TYPE((l.get(s...)-r.get(s...)))

template<int IL, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<MULTIPLIES>, Field<Geometry<TM, IL>, TL> const &l, TR const &r,
		TI ... s)
		ENABLE_IF_DECL_RET_TYPE((!is_field<TR>::value) ,(l.get(s...) * r))

template<int IR, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<MULTIPLIES>, TL const & l,
		Field<Geometry<TM, IR>, TR> const & r, TI ... s)
		ENABLE_IF_DECL_RET_TYPE((!is_field<TL>::value), (l * r.get(s...)))

template<int IL, int IR, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<MULTIPLIES>, Field<Geometry<TM, IL>, TL> const &l,
		Field<Geometry<TM, IR>, TR> const &r, TI ... s)
		DECL_RET_TYPE( (l.mesh.mapto(Int2Type<IL+IR>(),l,s...)*
						r.mesh.mapto(Int2Type<IL+IR>(),r,s...)) )

template<int IL, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<DIVIDES>, Field<Geometry<TM, IL>, TL> const &l, TR const &r,
		TI ... s)
		DECL_RET_TYPE( (l.get(s...)/ l.mesh.get(r,s...)) )

template<int I, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<DOT>, Field<Geometry<TM, I>, TL> const &l,
		Field<Geometry<TM, I>, TR> const &r, TI ... s)
		DECL_RET_TYPE((Dot(l.get(s...),r.get(s...))) )

template<int IL, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<DOT>, Field<Geometry<TM, IL>, TL> const &l,
		nTuple<3, TR> const &r, TI ... s)
		DECL_RET_TYPE((Dot(l.get(s...) , r)))

template<typename TL, int IR, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<DOT>, nTuple<3, TL> const & l,
		Field<Geometry<TM, IR>, TR> const & r, TI ... s)
		DECL_RET_TYPE((Dot(l , r.get(s...))))

template<int I, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<CROSS>, Field<Geometry<TM, I>, TL> const &l,
		Field<Geometry<TM, I>, TR> const &r, TI ... s)
		DECL_RET_TYPE( (Cross(l.get(s...),r.get(s...))))

template<int IL, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<CROSS>, Field<Geometry<TM, IL>, TL> const &l,
		nTuple<3, TR> const &r, TI ... s)
		DECL_RET_TYPE((Cross(l.get(s...) , r)))

template<typename TL, int IR, typename TR, typename TM, typename ...TI> inline auto OpEval(
		Int2Type<CROSS>, nTuple<3, TL> const & l,
		Field<Geometry<TM, IR>, TR> const & r, TI ... s)
		DECL_RET_TYPE((Cross(l , r.get(s...))))

// Check the availability of member function OpEval

template<int TOP, typename TM, typename TL, typename ...TI>
auto FieldUniOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TI ...s)
ENABLE_IF_DECL_RET_TYPE(
		(mesh_has_op<TM, TOP, TL>::value),
		(mesh.OpEval(Int2Type<TOP>(), l, s...))
)
;

template<int TOP, typename TM, typename TL, typename ...TI>
auto FieldUniOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TI ...s)
ENABLE_IF_DECL_RET_TYPE(
		(!mesh_has_op<TM, TOP, TL>::value),
		(OpEval(Int2Type<TOP>(), l, s...))
)
;

template<int TOP, typename TM, typename TL, typename TR, typename ...TI>
auto FieldBiOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r,
		TI ...s)
		ENABLE_IF_DECL_RET_TYPE(
				(mesh_has_op<TM, TOP, TL,TR>::value),
				(mesh.OpEval(Int2Type<TOP>(), l,r, s...))
		)
;

template<int TOP, typename TM, typename TL, typename TR, typename ...TI>
auto FieldBiOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r,
		TI ...s)
		ENABLE_IF_DECL_RET_TYPE(
				(!mesh_has_op<TM, TOP, TL,TR>::value),
				(OpEval(Int2Type<TOP>(), l,r, s...))
		)
;
}  // namespace fetl_impl

template<typename TM, int IL, int TOP, typename TL>
struct Field<Geometry<TM, IL>, UniOp<TOP, TL> >
{

public:

	typename ConstReferenceTraits<TL>::type l_;

	typedef Field<Geometry<TM, IL>, UniOp<TOP, TL> > this_type;
	TM const & mesh;

	enum
	{
		IForm = IL
	};

	Field(TL const & l) :
			mesh(l.mesh), l_(l)
	{
	}

	typedef decltype(fetl_impl::FieldUniOpEval(Int2Type<TOP>(), std::declval<TM>(),
					std::declval<TL>(),0)) value_type;

	template<typename ... TI> inline value_type get(TI ... s) const
	{
		return fetl_impl::FieldUniOpEval(Int2Type<TOP>(), mesh, l_, s...);
	}

};

template<typename TM, int IFORM, int TOP, typename TL, typename TR>
struct Field<Geometry<TM, IFORM>, BiOp<TOP, TL, TR> >
{

public:
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;

	TM const & mesh;
	typedef Field<Geometry<TM, IFORM>, BiOp<TOP, TL, TR> > this_type;
	enum
	{
		IForm = IFORM
	};

	Field(TL const & l, TR const & r) :
			mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}

	typedef decltype(fetl_impl::FieldBiOpEval(Int2Type<TOP>(),std::declval<TM>(),
					std::declval<TL>(),std::declval<TR>(),0)) value_type;

	template<typename ... TI> inline value_type get(TI ... s) const
	{
		return (fetl_impl::FieldBiOpEval(Int2Type<TOP>(), mesh, l_, r_, s...));
	}

private:

	template<int IL, typename VL, typename VR> static inline TM const & get_mesh(
			Field<Geometry<TM, IL>, VL> const & l, VR const & r)
	{
		return (l.mesh);
	}
	template<typename VL, int IR, typename VR> static inline TM const & get_mesh(
			VL const & l, Field<Geometry<TM, IR>, VR> const & r)
	{
		return (r.mesh);
	}

	template<int IL, typename VL, int IR, typename VR> static inline TM const & get_mesh(
			Field<Geometry<TM, IL>, VL> const & l,
			Field<Geometry<TM, IR>, VR> const & r)
	{
		return (l.mesh);
	}

}
;

template<typename TG, int IL, typename TL>
inline auto operator-(Field<Geometry<TG, IL>, TL> const & f)
DECL_RET_TYPE(
		( Field<Geometry<TG, IL>,
				UniOp<NEGATE,Field<Geometry<TG, IL>, TL> > > (f)))

template<typename TG, int IL, typename TL>
inline auto operator+(Field<Geometry<TG, IL>, TL> const & f)
DECL_RET_TYPE( f)

template<typename TGeo, typename TL, typename TR> inline auto //
operator+(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE(
		( Field<TGeo ,
				BiOp<PLUS,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))

template<typename TGeo, typename TL, typename TR> inline auto //
operator-(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE(
		( Field<TGeo ,
				BiOp<MINUS,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
operator*(Field<TG, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE((Field<TG,BiOp<MULTIPLIES,Field<TG,TL>,TR > > (lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
operator*(TL const & lhs, Field<TG, TR> const & rhs)
DECL_RET_TYPE((Field<TG,BiOp<MULTIPLIES,TL,Field<TG,TR> > > (lhs, rhs)))

template<typename TG, int IL, int IR, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, IR>, TR> const & rhs)
		DECL_RET_TYPE((Field<Geometry<TG,IL+IR>,BiOp<MULTIPLIES,
						Field<Geometry<TG,IL>,TL>,
						Field<Geometry<TG,IR>,TR> > > (lhs, rhs)))

template<typename TG, int IL, typename TL, typename TR> inline auto //
operator/(Field<Geometry<TG, IL>, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IL >,
				BiOp<DIVIDES,Field<Geometry<TG, IL>, TL>,TR > > (lhs, rhs))
)

/// *****************************************************************
/// To remve the ambiguity of operator define
template<typename TG, typename TL, int NR, typename TR> inline auto //
operator*(Field<TG, TL> const & lhs, nTuple<NR, TR> const & rhs)
DECL_RET_TYPE((Field<TG,
				BiOp<MULTIPLIES,Field<TG,TL>,nTuple<NR, TR> > > (lhs, rhs)))

template<typename TG, int NL, typename TL, typename TR> inline auto //
operator*(nTuple<NL, TL> const & lhs, Field<TG, TR> const & rhs)
DECL_RET_TYPE((Field<TG,
				BiOp<MULTIPLIES,nTuple<NL,TL>,Field<TG,TR> > > (lhs, rhs)))

template<typename TG, int IL, typename TL, int NR, typename TR> inline auto //
operator/(Field<Geometry<TG, IL>, TL> const & lhs, nTuple<NR, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IL >,
				BiOp<DIVIDES,Field<Geometry<TG, IL>, TL>,
				nTuple<NR,TR> > > (lhs, rhs))
)

/// *****************************************************************

template<typename TGeo, typename TL, typename TR> inline auto //
operator==(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((lhs-rhs))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, 0>, TL> const & lhs,
		Field<Geometry<TG, 0>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,0> ,
						BiOp<CROSS,Field<Geometry<TG, 0>, TL> ,
						Field<Geometry<TG, 0>, TR> > >(lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, 0>, TL> const & lhs, nTuple<3, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,0> ,
				BiOp<CROSS,Field<Geometry<TG, 0>, TL> ,
				nTuple<3,TR> > >(lhs, rhs)))

template<typename TG, typename TL, typename TR> inline auto //
Cross(nTuple<3, TL> const & lhs, Field<Geometry<TG, 0>, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,0> ,
				BiOp<CROSS,nTuple<3,TL> ,
				Field<Geometry<TG, 0>, TR> > >(lhs, rhs)))

template<typename TG, int IL, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, IL>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,IL> ,
						BiOp<DOT,Field<Geometry<TG, IL>, TL> ,
						Field<Geometry<TG, IL>, TR> > >(lhs, rhs)))

template<typename TG, typename TL, int IR, typename TR> inline auto //
Dot(nTuple<3, TL> const & lhs, Field<Geometry<TG, IR>, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IR> ,
				BiOp<DOT,nTuple<3,TL> ,
				Field<Geometry<TG, IR>, TR> > >(lhs, rhs)))

template<typename TG, int IL, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, IL>, TL> const & lhs, nTuple<3, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IL> ,
				BiOp<DOT,Field<Geometry<TG, IL>, TL> ,
				nTuple<3, TR> > >(lhs, rhs)))

template<typename TG, typename TR>
inline auto Grad(Field<Geometry<TG, 0>, TR> const & f)
DECL_RET_TYPE((Field<Geometry<TG, 1>,
				UniOp<GRAD,Field<Geometry<TG, 0>, TR> > >(f)))

template<typename TG, typename TR>
inline auto Diverge(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE((Field<Geometry<TG, 0>,
				UniOp<DIVERGE, Field<Geometry<TG, 1>, TR> > >( f)))

template<typename TG, typename TR>
inline auto Curl(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				UniOp<CURL, Field<Geometry<TG, 1>, TR> > >(f)))
template<typename TG, typename TR>
inline auto CurlPDX(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				UniOp<CURLPDX, Field<Geometry<TG, 1>, TR> > >(f)))
template<typename TG, typename TR>
inline auto CurlPDY(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				UniOp<CURLPDY, Field<Geometry<TG, 1>, TR> > >(f)))
template<typename TG, typename TR>
inline auto CurlPDZ(Field<Geometry<TG, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				UniOp<CURLPDZ, Field<Geometry<TG, 1>, TR> > >(f)))

template<typename TG, typename TR>
inline auto Curl(Field<Geometry<TG, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				UniOp<CURL, Field<Geometry<TG, 2>, TR> > >(f)))
template<typename TG, typename TR>
inline auto CurlPDX(Field<Geometry<TG, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				UniOp<CURLPDX, Field<Geometry<TG, 2>, TR> > >(f)))
template<typename TG, typename TR>
inline auto CurlPDY(Field<Geometry<TG, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				UniOp<CURLPDY, Field<Geometry<TG, 2>, TR> > >(f)))
template<typename TG, typename TR>
inline auto CurlPDZ(Field<Geometry<TG, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				UniOp<CURLPDZ, Field<Geometry<TG, 2>, TR> > >(f)))

template<typename TG, int IL, typename TL>
inline auto HodgeStar(Field<Geometry<TG, IL>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IL >= 0 && IL <= TG::NUM_OF_DIMS),
				Field<Geometry<TG, TG::NUM_OF_DIMS - IL>,
				UniOp<HODGESTAR
				,Field<Geometry<TG, IL>, TL> > >, Zero>::type(f)))

template<typename TG, int IL, typename TL>
inline auto d(Field<Geometry<TG, IL>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IL > 0 && IL+1 <= TG::NUM_OF_DIMS),
				Field<Geometry<TG, IL+1>,
				UniOp<EXTRIORDERIVATIVE,Field<Geometry<TG, IL>, TL> > >
				, Zero>::type(f)) )

template<typename TG, int IL, int IR, typename TL, typename TR>
inline auto Wedge(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, IR>, TR> const & rhs)
		DECL_RET_TYPE( ( Field<Geometry<TG,IL+IR> ,
						BiOp<WEDGE,Field<Geometry<TG, IL>, TL> ,
						Field<Geometry<TG, IR>, TR> > >
						(lhs, rhs)))

}
// namespace simpla
#endif /* OPERATIONS_H_ */
