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
#include "../utilities/type_utilites.h"

namespace simpla
{

template<int, typename > class nTuple;

namespace fetl_impl
{

template<int N, typename TL, typename TM, typename ...TI> inline auto OpEval(Int2Type<NEGATE>,
        Field<Geometry<TM, N>, TL> const & f, TI ... s)
        DECL_RET_TYPE((-f.get(s...)) )

} // namespace fetl_impl

template<typename TM, int IFORM, typename TL>
inline auto operator-(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(
		( Field<Geometry<TM, IFORM>,
				UniOp<NEGATE,Field<Geometry<TM, IFORM>, TL> > > (f)))

template<typename TM, int IFORM, typename TL>
inline auto operator+(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE( (f))

//****************************************************************************************************
namespace fetl_impl
{

template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<PLUS>,
        Field<Geometry<TM, IFORM>, TL> const &l, Field<Geometry<TM, IFORM>, TR> const &r, TI ... s)
        DECL_RET_TYPE((l.get(s...)+r.get(s...)))
}  // namespace fetl_impl

template<typename TGeo, typename TL, typename TR> inline auto //
operator+(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE( ( Field<TGeo , BiOp<PLUS,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))
//****************************************************************************************************
namespace fetl_impl
{
template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<MINUS>,
        Field<Geometry<TM, IFORM>, TL> const &l, Field<Geometry<TM, IFORM>, TR> const &r, TI ... s)
        DECL_RET_TYPE((l.get(s...)-r.get(s...)))

//template<typename TG, typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<MINUS>, TL const &l,
//        Field<TG, TR> const &r, TI ... s)
//        DECL_RET_TYPE((l -r.get(s...)))

template<typename TG, typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<MINUS>,
        Field<TG, TL> const &l, TR const &r, TI ... s)
        DECL_RET_TYPE((l.get(s...)-r))

}  // namespace fetl_impl
template<typename TGeo, typename TL, typename TR> inline auto //
operator-(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE( ( Field<TGeo , BiOp<MINUS,Field<TGeo, TL> , Field<TGeo, TR> > > (lhs, rhs)))

template<typename TGeo, typename TL, typename TR> inline auto //
operator-(Field<TGeo, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE( ( Field<TGeo , BiOp<MINUS,Field<TGeo, TL> , TR > > (lhs, rhs)))

//template<typename TGeo, typename TL, typename TR> inline auto //
//operator-(TL const & lhs, Field<TGeo, TR> const & rhs)
//DECL_RET_TYPE( ( Field<TGeo , BiOp<MINUS,TL , Field<TGeo, TR> > > (lhs, rhs)))

// *****************************************************************

namespace fetl_impl
{

template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<MULTIPLIES>,
        Field<Geometry<TM, IFORM>, TL> const &l, TR const &r, TI ... s)
        DECL_RET_TYPE((l.get(s...) * r))

template<int IR, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<MULTIPLIES>,
        TL const & l, Field<Geometry<TM, IR>, TR> const & r, TI ... s)
        DECL_RET_TYPE((l * r.get(s...)))

template<typename TM, int IL, typename TL, int IR, typename TR, typename ...TI> inline auto OpEval(Int2Type<MULTIPLIES>,
        Field<Geometry<TM, IL>, TL> const &l, nTuple<IR, TR> const &r, TI ... s)
        DECL_RET_TYPE((l.get(s...) * r))

template<typename TM, int IL, typename TL, int IR, typename TR, typename ...TI> inline auto OpEval(Int2Type<MULTIPLIES>,
        nTuple<IL, TL> const & l, Field<Geometry<TM, IR>, TR> const & r, TI ... s)
        DECL_RET_TYPE((l * r.get(s...)))

template<typename TM, int IL, int IR, typename TL, typename TR, typename ...TI> inline auto OpEval(Int2Type<MULTIPLIES>,
        Field<Geometry<TM, IL>, TL> const &l, Field<Geometry<TM, IR>, TR> const &r, TI ... s)
        DECL_RET_TYPE( (l.mesh.mapto(Int2Type<IL+IR>(),l,s...)*
				        r.mesh.mapto(Int2Type<IL+IR>(),r,s...)) )

}  // namespace fetl_impl

template<typename TM, int IL, int IR, typename TL, typename TR> inline auto //
operator*(Field<Geometry<TM, IL>, TL> const & lhs, Field<Geometry<TM, IR>, TR> const & rhs)
DECL_RET_TYPE((Field<Geometry<TM,IL+IR>,BiOp<MULTIPLIES,
				Field<Geometry<TM,IL>,TL>, Field<Geometry<TM,IR>,TR> > > (lhs, rhs)))

template<typename TGeo, typename TL, typename TR> inline auto //
operator*(Field<TGeo, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE((Field<TGeo,BiOp<MULTIPLIES,Field<TGeo,TL>,TR > > (lhs, rhs)))

template<typename TGeo, typename TL, typename TR> inline auto //
operator*(TL const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((Field<TGeo,BiOp<MULTIPLIES,TL,Field<TGeo,TR> > > (lhs, rhs)))
//
//// To remve the ambiguity of operator define
template<typename TG, typename TL, int NR, typename TR> inline auto //
operator*(Field<TG, TL> const & lhs, nTuple<NR, TR> const & rhs)
DECL_RET_TYPE((Field<TG,
				BiOp<MULTIPLIES,Field<TG,TL>,nTuple<NR, TR> > > (lhs, rhs)))

template<typename TG, int NL, typename TL, typename TR> inline auto //
operator*(nTuple<NL, TL> const & lhs, Field<TG, TR> const & rhs)
DECL_RET_TYPE((Field<TG,
				BiOp<MULTIPLIES,nTuple<NL,TL>,Field<TG,TR> > > (lhs, rhs)))

// *****************************************************************

namespace fetl_impl
{
template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<DIVIDES>,
        Field<Geometry<TM, IFORM>, TL> const &l, TR const &r, TI ... s)
        DECL_RET_TYPE( (l.get(s...)/ l.mesh.get(r,s...)) )
}  // namespace fetl_impl

template<typename TM, int IFORM, typename TL, typename TR> inline auto //
operator/(Field<Geometry<TM, IFORM>, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TM,IFORM >,
				BiOp<DIVIDES,Field<Geometry<TM, IFORM>, TL>,TR > > (lhs, rhs))
)

template<typename TG, int IFORM, typename TL, int NR, typename TR> inline auto //
operator/(Field<Geometry<TG, IFORM>, TL> const & lhs, nTuple<NR, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IFORM >,
				BiOp<DIVIDES,Field<Geometry<TG, IFORM>, TL>,
				nTuple<NR,TR> > > (lhs, rhs))
)

//****************************************************************************************************
template<typename TGeo, typename TL, typename TR> inline auto //
operator==(Field<TGeo, TL> const & lhs, Field<TGeo, TR> const & rhs)
DECL_RET_TYPE((lhs-rhs))
//****************************************************************************************************

namespace fetl_impl
{

template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<CROSS>,
        Field<Geometry<TM, IFORM>, TL> const &l, Field<Geometry<TM, IFORM>, TR> const &r, TI ... s)
        DECL_RET_TYPE( (Cross(l.get(s...),r.get(s...))))

template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<CROSS>,
        Field<Geometry<TM, IFORM>, TL> const &l, nTuple<3, TR> const &r, TI ... s)
        DECL_RET_TYPE((Cross(l.get(s...) , r)))

template<typename TL, int IR, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<CROSS>,
        nTuple<3, TL> const & l, Field<Geometry<TM, IR>, TR> const & r, TI ... s)
        DECL_RET_TYPE((Cross(l , r.get(s...))))

} // namespace fetl_impl
template<typename TG, typename TL, typename TR> inline auto //
Cross(Field<Geometry<TG, 0>, TL> const & lhs, Field<Geometry<TG, 0>, TR> const & rhs)
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

//****************************************************************************************************
namespace fetl_impl
{

template<int I, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<DOT>,
        Field<Geometry<TM, I>, TL> const &l, Field<Geometry<TM, I>, TR> const &r, TI ... s)
        DECL_RET_TYPE((Dot(l.get(s...),r.get(s...))) )

template<int IFORM, typename TL, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<DOT>,
        Field<Geometry<TM, IFORM>, TL> const &l, nTuple<3, TR> const &r, TI ... s)
        DECL_RET_TYPE((Dot(l.get(s...) , r)))

template<typename TL, int IR, typename TR, typename TM, typename ...TI> inline auto OpEval(Int2Type<DOT>,
        nTuple<3, TL> const & l, Field<Geometry<TM, IR>, TR> const & r, TI ... s)
        DECL_RET_TYPE((Dot(l , r.get(s...))))

}  // namespace fetl_impl
//template<typename TG, int IFORM, typename TL, typename TR> inline auto //
//Dot(Field<Geometry<TG, IFORM>, TL> const & lhs,
//		Field<Geometry<TG, IFORM>, TR> const & rhs)
//		DECL_RET_TYPE(
//				(Field<Geometry<TG,IFORM> ,
//						BiOp<DOT,Field<Geometry<TG, IFORM>, TL> ,
//						Field<Geometry<TG, IFORM>, TR> > >(lhs, rhs)))

template<typename TG, typename TL, int IR, typename TR> inline auto //
Dot(nTuple<3, TL> const & lhs, Field<Geometry<TG, IR>, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IR> ,
				BiOp<DOT,nTuple<3,TL> ,
				Field<Geometry<TG, IR>, TR> > >(lhs, rhs)))

template<typename TG, int IFORM, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TG, IFORM>, TL> const & lhs, nTuple<3, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TG,IFORM> ,
				BiOp<DOT,Field<Geometry<TG, IFORM>, TL> ,
				nTuple<3, TR> > >(lhs, rhs)))

template<typename TM, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TM, 1>, TL> const & lhs, Field<Geometry<TM, 1>, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TM,0> ,
				BiOp<DOT,Field<Geometry<TM, 1>, TL> ,
				Field<Geometry<TM, 1>, TR> > >(lhs, rhs)))

template<typename TM, typename TL, typename TR> inline auto //
Dot(Field<Geometry<TM, 2>, TL> const & lhs, Field<Geometry<TM, 2>, TR> const & rhs)
DECL_RET_TYPE(
		(Field<Geometry<TM,0> ,
				BiOp<DOT,Field<Geometry<TM, 2>, TL> ,
				Field<Geometry<TM, 2>, TR> > >(lhs, rhs)))

//****************************************************************************************************
template<typename TM, typename TR>
inline auto Grad(Field<Geometry<TM, 0>, TR> const & f)
DECL_RET_TYPE((Field<Geometry<TM, 1>,
				UniOp<GRAD,Field<Geometry<TM, 0>, TR> > >(f)))
//****************************************************************************************************
template<typename TM, typename TR>
inline auto Diverge(Field<Geometry<TM, 1>, TR> const & f)
DECL_RET_TYPE((Field<Geometry<TM, 0>,
				UniOp<DIVERGE, Field<Geometry<TM, 1>, TR> > >( f)))
//****************************************************************************************************
template<typename TM, typename TR>
inline auto Curl(Field<Geometry<TM, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 2>,
				UniOp<CURL, Field<Geometry<TM, 1>, TR> > >(f)))
template<typename TM, typename TR>
inline auto CurlPDX(Field<Geometry<TM, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 2>,
				UniOp<CURLPDX, Field<Geometry<TM, 1>, TR> > >(f)))
template<typename TM, typename TR>
inline auto CurlPDY(Field<Geometry<TM, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 2>,
				UniOp<CURLPDY, Field<Geometry<TM, 1>, TR> > >(f)))
template<typename TM, typename TR>
inline auto CurlPDZ(Field<Geometry<TM, 1>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 2>,
				UniOp<CURLPDZ, Field<Geometry<TM, 1>, TR> > >(f)))

template<typename TM, typename TR>
inline auto Curl(Field<Geometry<TM, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 1>,
				UniOp<CURL, Field<Geometry<TM, 2>, TR> > >(f)))
template<typename TM, typename TR>
inline auto CurlPDX(Field<Geometry<TM, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 1>,
				UniOp<CURLPDX, Field<Geometry<TM, 2>, TR> > >(f)))
template<typename TM, typename TR>
inline auto CurlPDY(Field<Geometry<TM, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 1>,
				UniOp<CURLPDY, Field<Geometry<TM, 2>, TR> > >(f)))
template<typename TM, typename TR>
inline auto CurlPDZ(Field<Geometry<TM, 2>, TR> const & f)
DECL_RET_TYPE( (Field<Geometry<TM, 1>,
				UniOp<CURLPDZ, Field<Geometry<TM, 2>, TR> > >(f)))
//****************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto HodgeStar(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IFORM >= 0 && IFORM <= TM::NUM_OF_DIMS),
				Field<Geometry<TM, TM::NUM_OF_DIMS - IFORM>,
				UniOp<HODGESTAR
				,Field<Geometry<TM, IFORM>, TL> > >, Zero>::type(f)))
//****************************************************************************************************
template<typename TM, int IFORM, typename TL>
inline auto d(Field<Geometry<TM, IFORM>, TL> const & f)
DECL_RET_TYPE(
		(typename std::conditional<(IFORM > 0 && IFORM+1 <= TM::NUM_OF_DIMS),
				Field<Geometry<TM, IFORM+1>,
				UniOp<EXTRIORDERIVATIVE,Field<Geometry<TM, IFORM>, TL> > >
				, Zero>::type(f)) )
//****************************************************************************************************
template<typename TM, int IFORM, int IR, typename TL, typename TR>
inline auto Wedge(Field<Geometry<TM, IFORM>, TL> const & lhs, Field<Geometry<TM, IR>, TR> const & rhs)
DECL_RET_TYPE( ( Field<Geometry<TM,IFORM+IR> ,
				BiOp<WEDGE,Field<Geometry<TM, IFORM>, TL> ,
				Field<Geometry<TM, IR>, TR> > >
				(lhs, rhs)))

namespace fetl_impl
{

// Check the availability of member function OpEval

HAS_MEMBER_FUNCTION(OpEval)

template<int TOP, typename TM, typename TL, typename ...TI>
auto FieldUniOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TI ...s)
ENABLE_IF_DECL_RET_TYPE(
		(has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TI...>::value),
		(mesh.OpEval(Int2Type<TOP>(), l, s...))
)
;

template<int TOP, typename TM, typename TL, typename ...TI>
auto FieldUniOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TI ...s)
ENABLE_IF_DECL_RET_TYPE(
		(!has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TI...>::value),
		(OpEval(Int2Type<TOP>(), l, s...))
)
;

template<int TOP, typename TM, typename TL, typename TR, typename ...TI>
auto FieldBiOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r, TI ...s)
ENABLE_IF_DECL_RET_TYPE(
		(has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TR const &,TI...>::value),
		(mesh.OpEval(Int2Type<TOP>(), l,r, s...))
)
;

template<int TOP, typename TM, typename TL, typename TR, typename ...TI>
auto FieldBiOpEval(Int2Type<TOP>, TM const & mesh, TL const &l, TR const &r, TI ...s)
ENABLE_IF_DECL_RET_TYPE(
		(!has_member_function_OpEval<TM, Int2Type<TOP>, TL const &,TR const &,TI...>::value),
		(OpEval(Int2Type<TOP>(), l,r, s...))
)
;

}  // namespace fetl_impl

template<typename TM, int IFORM, int TOP, typename TL>
struct Field<Geometry<TM, IFORM>, UniOp<TOP, TL> >
{

public:

	typename ConstReferenceTraits<TL>::type l_;

	typedef Field<Geometry<TM, IFORM>, UniOp<TOP, TL> > this_type;
	TM const & mesh;

	enum
	{
		IForm = IFORM
	};

	Field(TL const & l)
			: mesh(l.mesh), l_(l)
	{
	}

	size_t size() const
	{
		return mesh.GetNumOfElements(IFORM);
	}
	typedef decltype(fetl_impl::FieldUniOpEval(Int2Type<TOP>(), std::declval<TM>(),
					std::declval<TL>(),0)) value_type;

	template<typename ... TI> inline value_type get(TI ... s) const
	{
		return fetl_impl::FieldUniOpEval(Int2Type<TOP>(), mesh, l_, s...);
	}

};

template<typename TGeo, int TOP, typename TL, typename TR>
struct Field<TGeo, BiOp<TOP, TL, TR> >
{

public:
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;
	typedef TGeo geometry_type;
	typedef typename geometry_type::mesh_type mesh_type;

	typedef Field<geometry_type, BiOp<TOP, TL, TR> > this_type;

	static constexpr int IForm = geometry_type::IForm;

	mesh_type const & mesh;

	Field(TL const & l, TR const & r)
			: mesh(get_mesh(l, r)), l_(l), r_(r)
	{
	}
	size_t size() const
	{
		return mesh.GetNumOfElements(IForm);
	}
private:
	typedef typename std::remove_cv<typename std::remove_reference<TL>::type>::type L_type;
	typedef typename std::remove_cv<typename std::remove_reference<TR>::type>::type R_type;
public:
	typedef decltype(fetl_impl::FieldBiOpEval(Int2Type<TOP>(),std::declval<mesh_type>(),
					std::declval<L_type>(),std::declval<R_type>(),0)) value_type;

	template<typename ... TI> inline value_type get(TI ... s) const
	{
		return (fetl_impl::FieldBiOpEval(Int2Type<TOP>(), mesh, l_, r_, s...));
	}

private:

	template<int IL, typename VL, typename VR> static inline mesh_type const &
	get_mesh(Field<Geometry<mesh_type, IL>, VL> const & l, VR const & r)
	{
		return (l.mesh);
	}
	template<typename VL, int IR, typename VR> static inline mesh_type const &
	get_mesh(VL const & l, Field<Geometry<mesh_type, IR>, VR> const & r)
	{
		return (r.mesh);
	}

	template<int IL, typename VL, int IR, typename VR> static inline mesh_type const &
	get_mesh(Field<Geometry<mesh_type, IL>, VL> const & l, Field<Geometry<mesh_type, IR>, VR> const & r)
	{
		return (l.mesh);
	}

}
;

}
// namespace simpla
#endif /* OPERATIONS_H_ */
