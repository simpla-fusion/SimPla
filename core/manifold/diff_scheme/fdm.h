/*
 * fdm.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef FDM_H_
#define FDM_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../../utilities/sp_type_traits.h"
#include "../../physics/constants.h"
#include "../manifold.h"
#include "../calculus.h"
#include "../../field/field.h"
namespace simpla
{

template<typename ... > class _Field;
template<typename, size_t> class Domain;

/** \ingroup DiffScheme
 *  \brief template of FvMesh
 */
template<typename G>
struct FiniteDiffMethod
{
	typedef FiniteDiffMethod<G> this_type;
	typedef G geometry_type;
	typedef typename geometry_type::topology_type topology_type;
	typedef typename geometry_type::compact_index_type compact_index_type;
	typedef Real scalar_type;
	static constexpr size_t NUM_OF_COMPONENT_TYPE = G::ndims + 1;
	static constexpr size_t ndims = G::ndims;

	G const * geo;
	FiniteDiffMethod() :
			geo(nullptr)
	{
	}
	FiniteDiffMethod(G const * g) :
			geo(g)
	{
	}
	FiniteDiffMethod(this_type const & r) :
			geo(r.geo)
	{
	}
	~FiniteDiffMethod() = default;

	this_type & operator=(this_type const &) = delete;

	void geometry(G const*g)
	{
		geo = g;
	}
	G const &geometry() const
	{
		return *geo;
	}
//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************

//	template<typename T, typename ...Others>
//	T const & calculate(T const & v, Others &&... s) const;
//
//	template<typename ...T, typename ...Others>
//	inline typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type
//	calculate(nTuple<Expression<T...>> const & v, Others &&... s) const;
//
//	template<typename TC, typename TD, typename Others>
//	inline typename _Field<TC, TD>::value_type const &
//	calculate( _Field<TC, TD> const &f, Others s) const;
//
//	template<typename TOP, typename TL, typename ...Others>
//	inline typename field_result_of<TOP(TL,Others ... )>::type
//	calculate(_Field<Expression<TOP, TL>> const &f, Others &&... s) const;
//
//	template<typename TOP, typename TL, typename TR, typename ...Others>
//	inline typename field_result_of<TOP(TL,TR,Others ... )>::type
//	calculate(_Field<Expression<TOP, TL, TR>> const &f, Others &&... s) const;
//

	template<typename ...Others>
	Real calculate(Real v, Others &&... s) const
	{
		return v;
	}

	template<typename ...Others>
	int calculate(int v, Others &&... s) const
	{
		return v;
	}

	template<typename ...Others>
	std::complex<Real> calculate(std::complex<Real> v, Others &&... s) const
	{
		return v;
	}

	template<typename ...T, typename ...Others>
	inline typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type
	calculate(nTuple<Expression<T...>> const & v, Others &&... s) const
	{
		typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type res;
		res=v;
		return std::move(res);
	}

	template<typename TC, typename TD, typename ... Others>
	inline typename field_traits<_Field<TC, TD> >::value_type
	calculate(_Field<TC, TD> const &f, Others && ... s) const
	{
		return f.get(std::forward<Others>(s)...);
	}

	template<typename TOP, typename TL, typename TR, typename ...Others>
	inline typename field_traits< _Field<Expression<TOP, TL, TR>>>::value_type
	calculate(_Field<Expression<TOP, TL, TR>> const &f, Others &&... s) const
	{
		return f.op_(calculate(f.lhs,std::forward<Others>(s)...),
				calculate(f.rhs,std::forward<Others>(s)...));
	}

	template<typename TOP, typename TL, typename ...Others>
	inline typename field_traits< _Field<Expression<TOP, TL,std::nullptr_t>>>::value_type
	calculate(_Field<Expression<TOP, TL,std::nullptr_t>> const &f, Others &&... s) const
	{
		return f.op_(calculate(f.lhs,std::forward<Others>(s)...) );
	}

	template<typename T,typename TI>
	inline typename field_traits<_Field<_impl::ExteriorDerivative< VERTEX,T> >>::value_type
	calculate(_Field<_impl::ExteriorDerivative<VERTEX,T> > const & f, TI s) const
	{
		auto D = geo->delta_index(s);

		return (calculate(f.lhs, s + D) * geo->volume(s + D)
				- calculate(f.lhs, s - D) * geo->volume(s - D)) * geo->inv_volume(s);
	}

	template<typename T,typename TI>
	inline typename field_traits<_Field<_impl::ExteriorDerivative< EDGE,T> >>::value_type
	calculate(_Field<_impl::ExteriorDerivative<EDGE,T> > const & expr, TI s) const
	{
		auto const & f=expr.lhs;

		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return (

				(

						calculate(f, s + Y) * geo->volume(s + Y) //
						- calculate(f, s - Y) * geo->volume(s - Y)//

				) - (

						calculate(f, s + Z) * geo->volume(s + Z)//
						- calculate(f, s - Z) * geo->volume(s - Z)//

				)

		) * geo->inv_volume(s);

	}

	template<typename T,typename TI>
	inline typename field_traits<_Field<_impl::ExteriorDerivative< FACE,T> >>::value_type
	calculate(_Field<_impl::ExteriorDerivative<FACE,T> > const & expr, TI s) const
	{
		auto const & f=expr.lhs;
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return (

				calculate(f, s + X) * geo->volume(s + X)

				- calculate(f, s - X) * geo->volume(s - X) //
				+ calculate(f, s + Y) * geo->volume(s + Y)//
				- calculate(f, s - Y) * geo->volume(s - Y)//
				+ calculate(f, s + Z) * geo->volume(s + Z)//
				- calculate(f, s - Z) * geo->volume(s - Z)//

		) * geo->inv_volume(s)

		;
	}
//
////	template<typename TM, size_t IL, typename TL> void calculate(
////			_impl::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
////			compact_index_type s) const = delete;
////
////	template<typename TM, size_t IL, typename TL> void calculate(
////			_impl::CodifferentialDerivative,
////			_Field<TL...> const & f, compact_index_type s) const = delete;

	template<typename T >
	inline typename field_traits<_Field< _impl::CodifferentialDerivative< EDGE, T> > >::value_type
	calculate (_Field<_impl::CodifferentialDerivative< EDGE, T>> const & expr,
			compact_index_type s) const
	{
		auto const & f=expr.lhs;

		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		-(

				calculate(f, s + X) * geo->dual_volume(s + X)

				- calculate(f, s - X) * geo->dual_volume(s - X)

				+ calculate(f, s + Y) * geo->dual_volume(s + Y)

				- calculate(f, s - Y) * geo->dual_volume(s - Y)

				+ calculate(f, s + Z) * geo->dual_volume(s + Z)

				- calculate(f, s - Z) * geo->dual_volume(s - Z)

		) * geo->inv_dual_volume(s)

		;

	}

	template<typename T >
	inline typename field_traits<_Field< _impl::CodifferentialDerivative< FACE, T> > >::value_type
	calculate (_Field<_impl::CodifferentialDerivative< FACE, T>> const & expr,
			compact_index_type s) const
	{
		auto const & f=expr.lhs;
		auto X = geo->delta_index(s);
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return

		-(
				(calculate(f, s + Y) * (geo->dual_volume(s + Y))
						- calculate(f, s - Y) * (geo->dual_volume(s - Y)))

				- (calculate(f, s + Z) * (geo->dual_volume(s + Z))
						- calculate(f, s - Z) * (geo->dual_volume(s - Z)))

		) * geo->inv_dual_volume(s)

		;
	}

	template<typename T >
	inline typename field_traits<_Field< _impl::CodifferentialDerivative< VOLUME, T> > >::value_type
	calculate (_Field<_impl::CodifferentialDerivative< VOLUME, T>> const & expr,
			compact_index_type s) const
	{
		auto const & f=expr.lhs;
		auto D = geo->delta_index(geo->dual(s));
		return

		-(

				calculate(f, s + D) * (geo->dual_volume(s + D)) //
				- calculate(f, s - D) * (geo->dual_volume(s - D))

		) * geo->inv_dual_volume(s)

		;
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,VERTEX,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<VERTEX,VERTEX,TL,TR>> const & expr,
			compact_index_type s) const
	{
		return (calculate(expr.lhs, s) * calculate(expr.rhs, s));
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,EDGE,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<VERTEX,EDGE,TL,TR>> const & expr,
			compact_index_type s) const
	{
		auto X = geo->delta_index(s);

		return (calculate(expr.lhs, s - X) + calculate(expr.lhs, s + X)) * 0.5
		* calculate(expr.rhs, s);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,FACE,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<VERTEX,FACE,TL,TR>> const & expr,
			compact_index_type s) const
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return (

				calculate(l, (s - Y) - Z) +

				calculate(l, (s - Y) + Z) +

				calculate(l, (s + Y) - Z) +

				calculate(l, (s + Y) + Z)

		) * 0.25 * calculate(r, s);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,VOLUME,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<VERTEX,VOLUME,TL,TR>> const & expr,
			compact_index_type s) const
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return (

				calculate(l, ((s - X) - Y) - Z) +

				calculate(l, ((s - X) - Y) + Z) +

				calculate(l, ((s - X) + Y) - Z) +

				calculate(l, ((s - X) + Y) + Z) +

				calculate(l, ((s + X) - Y) - Z) +

				calculate(l, ((s + X) - Y) + Z) +

				calculate(l, ((s + X) + Y) - Z) +

				calculate(l, ((s + X) + Y) + Z)

		) * 0.125 * calculate(r, s);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<EDGE,VERTEX,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<EDGE,VERTEX,TL,TR>> const & expr,
			compact_index_type s) const
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = geo->delta_index(s);
		return calculate(l, s) * (calculate(r, s - X) + calculate(r, s + X))
		* 0.5;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<EDGE,EDGE,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<EDGE,EDGE,TL,TR>> const & expr,
			compact_index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto Y = geo->delta_index(geo->roate(geo->dual(s)));
		auto Z = geo->delta_index(geo->inverse_roate(geo->dual(s)));

		return ((calculate(l, s - Y) + calculate(l, s + Y))
				* (calculate(l, s - Z) + calculate(l, s + Z)) * 0.25);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<EDGE,FACE,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<EDGE,FACE,TL,TR>> const & expr,
			compact_index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		(

				(calculate(l, (s - Y) - Z) + calculate(l, (s - Y) + Z)
						+ calculate(l, (s + Y) - Z) + calculate(l, (s + Y) + Z))
				* (calculate(r, s - X) + calculate(r, s + X))
				+

				(calculate(l, (s - Z) - X) + calculate(l, (s - Z) + X)
						+ calculate(l, (s + Z) - X)
						+ calculate(l, (s + Z) + X))
				* (calculate(r, s - Y) + calculate(r, s + Y))
				+

				(calculate(l, (s - X) - Y) + calculate(l, (s - X) + Y)
						+ calculate(l, (s + X) - Y)
						+ calculate(l, (s + X) + Y))
				* (calculate(r, s - Z) + calculate(r, s + Z))

		) * 0.125;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<FACE,VERTEX,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<FACE,VERTEX,TL,TR>> const & expr,
			compact_index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto Y = geo->delta_index(geo->roate(geo->dual(s)));
		auto Z = geo->delta_index(geo->inverse_roate(geo->dual(s)));

		return calculate(l, s)
		* (calculate(r, (s - Y) - Z) + calculate(r, (s - Y) + Z)
				+ calculate(r, (s + Y) - Z)
				+ calculate(r, (s + Y) + Z)) * 0.25;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<FACE,EDGE,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<FACE,EDGE,TL,TR>> const & expr,
			compact_index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		(

				(calculate(r, (s - Y) - Z) + calculate(r, (s - Y) + Z)
						+ calculate(r, (s + Y) - Z) + calculate(r, (s + Y) + Z))
				* (calculate(l, s - X) + calculate(l, s + X))

				+ (calculate(r, (s - Z) - X) + calculate(r, (s - Z) + X)
						+ calculate(r, (s + Z) - X)
						+ calculate(r, (s + Z) + X))
				* (calculate(l, s - Y) + calculate(l, s + Y))

				+ (calculate(r, (s - X) - Y) + calculate(r, (s - X) + Y)
						+ calculate(r, (s + X) - Y)
						+ calculate(r, (s + X) + Y))
				* (calculate(l, s - Z) + calculate(l, s + Z))

		) * 0.125;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VOLUME,VERTEX,TL,TR> > >::value_type
	calculate (_Field<_impl::Wedge<VOLUME,VERTEX,TL,TR>> const & expr,
			compact_index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		calculate(l, s) * (

				calculate(r, ((s - X) - Y) - Z) + //
				calculate(r, ((s - X) - Y) + Z) +//
				calculate(r, ((s - X) + Y) - Z) +//
				calculate(r, ((s - X) + Y) + Z) +//
				calculate(r, ((s + X) - Y) - Z) +//
				calculate(r, ((s + X) - Y) + Z) +//
				calculate(r, ((s + X) + Y) - Z) +//
				calculate(r, ((s + X) + Y) + Z)//

		) * 0.125;
	}
//
////***************************************************************************************************
//
//	template<typename TM, size_t IL, typename TL> inline auto calculate(
//			_impl::HodgeStar, _Field<Domain<TM, IL>, TL> const & f,
//			compact_index_type s) const -> typename std::remove_reference<decltype(calculate(f,s))>::type
//	{
////		auto X = geo->DI(0,s);
////		auto Y = geo->DI(1,s);
////		auto Z =geo->DI(2,s);
////
////		return
////
////		(
////
////		calculate(f,((s + X) - Y) - Z)*geo->inv_volume(((s + X) - Y) - Z) +
////
////		calculate(f,((s + X) - Y) + Z)*geo->inv_volume(((s + X) - Y) + Z) +
////
////		calculate(f,((s + X) + Y) - Z)*geo->inv_volume(((s + X) + Y) - Z) +
////
////		calculate(f,((s + X) + Y) + Z)*geo->inv_volume(((s + X) + Y) + Z) +
////
////		calculate(f,((s - X) - Y) - Z)*geo->inv_volume(((s - X) - Y) - Z) +
////
////		calculate(f,((s - X) - Y) + Z)*geo->inv_volume(((s - X) - Y) + Z) +
////
////		calculate(f,((s - X) + Y) - Z)*geo->inv_volume(((s - X) + Y) - Z) +
////
////		calculate(f,((s - X) + Y) + Z)*geo->inv_volume(((s - X) + Y) + Z)
////
////		) * 0.125 * geo->volume(s);
//
//		return calculate(f, s) /** geo->_impl::HodgeStarVolumeScale(s)*/;
//	}
//
//	template<typename TM, typename TL, typename TR> void calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, VERTEX>, TL> const & f,
//			compact_index_type s) const = delete;
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, EDGE>, TL> const & f,
//			compact_index_type s) const ->decltype(calculate(f,s)*v[0])
//	{
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return
//
//		(calculate(f, s + X) - calculate(f, s - X)) * 0.5 * v[0] //
//		+ (calculate(f, s + Y) - calculate(f, s - Y)) * 0.5 * v[1] //
//		+ (calculate(f, s + Z) - calculate(f, s - Z)) * 0.5 * v[2];
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, FACE>, TL> const & f,
//			compact_index_type s) const ->decltype(calculate(f,s)*v[0])
//	{
//		size_t n = geo->component_number(s);
//
//		auto X = geo->delta_index(s);
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//		return
//
//		(calculate(f, s + Y) + calculate(f, s - Y)) * 0.5 * v[(n + 2) % 3] -
//
//		(calculate(f, s + Z) + calculate(f, s - Z)) * 0.5 * v[(n + 1) % 3];
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, VOLUME>, TL> const & f,
//			compact_index_type s) const ->decltype(calculate(f,s)*v[0])
//	{
//		size_t n = geo->component_number(geo->dual(s));
//		size_t D = geo->delta_index(geo->dual(s));
//
//		return (calculate(f, s + D) - calculate(f, s - D)) * 0.5 * v[n];
//	}
//
////**************************************************************************************************
//// Non-standard operation
//// For curlpdx
//
//	template<typename TM, size_t N, typename TL> inline auto calculate(
//			_impl::ExteriorDerivative, _Field<Domain<TM, EDGE>, TL> const & f,
//			std::integral_constant<size_t, N>,
//			compact_index_type s) const -> decltype(calculate(f,s)-calculate(f,s))
//	{
//
//		auto X = geo->delta_index(geo->dual(s));
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//
//		Y = (geo->component_number(Y) == N) ? Y : 0UL;
//		Z = (geo->component_number(Z) == N) ? Z : 0UL;
//
//		return (calculate(f, s + Y) - calculate(f, s - Y))
//				- (calculate(f, s + Z) - calculate(f, s - Z));
//	}
//
//	template<typename ...TL, size_t N> inline auto calculate(
//			_impl::CodifferentialDerivative, _Field<TL...> const & f,
//			std::integral_constant<size_t, N>,
//			compact_index_type s) const ->
//			typename std::enable_if<field_traits<_Field<TL...>>::iform==FACE,
//			decltype((calculate(f,s)-calculate(f,s))*std::declval<scalar_type>())>::type
//	{
//
//		auto X = geo->delta_index(s);
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//
//		Y = (geo->component_number(Y) == N) ? Y : 0UL;
//		Z = (geo->component_number(Z) == N) ? Z : 0UL;
//
//		return (
//
//		calculate(f, s + Y) * (geo->dual_volume(s + Y))      //
//		- calculate(f, s - Y) * (geo->dual_volume(s - Y))    //
//		- calculate(f, s + Z) * (geo->dual_volume(s + Z))    //
//		+ calculate(f, s - Z) * (geo->dual_volume(s - Z))    //
//
//		) * geo->inv_dual_volume(s);
//	}
//	template<typename TM, size_t IL, typename TR> inline auto calculate(
//			_impl::MapTo, std::integral_constant<size_t, IL> const &,
//			_Field<Domain<TM, IL>, TR> const & f, compact_index_type s) const
//			DECL_RET_TYPE(calculate(f,s))
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, VERTEX> const &,
//			_Field<Domain<TM, EDGE>, TR> const & f,
//			compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(calculate(f,s))>::type,3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(calculate(f,s))>::type,
//				3>(
//		{
//
//		(calculate(f, s - X) + calculate(f, s + X)) * 0.5, //
//		(calculate(f, s - Y) + calculate(f, s + Y)) * 0.5, //
//		(calculate(f, s - Z) + calculate(f, s + Z)) * 0.5
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, EDGE> const &,
//			_Field<Domain<TM, VERTEX>, TR> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(calculate(f,s)[0])>::type
//	{
//
//		auto n = geo->component_number(s);
//		auto D = geo->delta_index(s);
//
//		return ((calculate(f, s - D)[n] + calculate(f, s + D)[n]) * 0.5);
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, VERTEX> const &,
//			_Field<Domain<TM, FACE>, TR> const & f,
//			compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(calculate(f,s))>::type,3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(calculate(f,s))>::type,
//				3>(
//		{ (
//
//		calculate(f, (s - Y) - Z) +
//
//		calculate(f, (s - Y) + Z) +
//
//		calculate(f, (s + Y) - Z) +
//
//		calculate(f, (s + Y) + Z)
//
//		) * 0.25,
//
//		(
//
//		calculate(f, (s - Z) - X) +
//
//		calculate(f, (s - Z) + X) +
//
//		calculate(f, (s + Z) - X) +
//
//		calculate(f, (s + Z) + X)
//
//		) * 0.25,
//
//		(
//
//		calculate(f, (s - X) - Y) +
//
//		calculate(f, (s - X) + Y) +
//
//		calculate(f, (s + X) - Y) +
//
//		calculate(f, (s + X) + Y)
//
//		) * 0.25
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, FACE> const &,
//			_Field<Domain<TM, VERTEX>, TR> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(calculate(f,s)[0])>::type
//	{
//
//		auto n = geo->component_number(geo->dual(s));
//		auto X = geo->delta_index(geo->dual(s));
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//
//		return (
//
//		(
//
//		calculate(f, (s - Y) - Z)[n] +
//
//		calculate(f, (s - Y) + Z)[n] +
//
//		calculate(f, (s + Y) - Z)[n] +
//
//		calculate(f, (s + Y) + Z)[n]
//
//		) * 0.25
//
//		);
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, VOLUME>,
//			_Field<Domain<TM, FACE>, TR> const & f,
//			compact_index_type s) const ->nTuple<decltype(calculate(f,s) ),3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(calculate(f,s))>::type,
//				3>(
//		{
//
//		(calculate(f, s - X) + calculate(f, s + X)) * 0.5, //
//		(calculate(f, s - Y) + calculate(f, s + Y)) * 0.5, //
//		(calculate(f, s - Z) + calculate(f, s + Z)) * 0.5
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, FACE>,
//			_Field<Domain<TM, VOLUME>, TR> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(calculate(f,s)[0])>::type
//	{
//
//		auto n = geo->component_number(geo->dual(s));
//		auto D = geo->delta_index(geo->dual(s));
//
//		return ((calculate(f, s - D)[n] + calculate(f, s + D)[n]) * 0.5);
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, VOLUME>,
//			_Field<Domain<TM, EDGE>, TR> const & f,
//			compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(calculate(f,s) )>::type,3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(calculate(f,s))>::type,
//				3>(
//		{ (
//
//		calculate(f, (s - Y) - Z) +
//
//		calculate(f, (s - Y) + Z) +
//
//		calculate(f, (s + Y) - Z) +
//
//		calculate(f, (s + Y) + Z)
//
//		) * 0.25,
//
//		(
//
//		calculate(f, (s - Z) - X) +
//
//		calculate(f, (s - Z) + X) +
//
//		calculate(f, (s + Z) - X) +
//
//		calculate(f, (s + Z) + X)
//
//		) * 0.25,
//
//		(
//
//		calculate(f, (s - X) - Y) +
//
//		calculate(f, (s - X) + Y) +
//
//		calculate(f, (s + X) - Y) +
//
//		calculate(f, (s + X) + Y)
//
//		) * 0.25,
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
//			std::integral_constant<size_t, EDGE>,
//			_Field<TR, Domain<TM, VOLUME>> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(calculate(f,s)[0])>::type
//	{
//
//		auto n = geo->component_number(geo->dual(s));
//		auto X = geo->delta_index(geo->dual(s));
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//		return (
//
//		(
//
//		calculate(f, (s - Y) - Z)[n] +
//
//		calculate(f, (s - Y) + Z)[n] +
//
//		calculate(f, (s + Y) - Z)[n] +
//
//		calculate(f, (s + Y) + Z)[n]
//
//		) * 0.25
//
//		);
//	}

public:

private:

//	template<typename TL>
//	auto calculate(TL const& f, compact_index_type s) const
//	DECL_RET_TYPE((get_value(f,s)) )
//
//	template<typename TOP, typename TL, template<typename > class F>
//	auto calculate(F<Expression<TOP, TL>> const& f,
//			compact_index_type s) const
//					DECL_RET_TYPE(this->calculate(f.op_,
//									integer_sequence<size_t,field_traits<TL>::iform>(),f.lhs,s ) )
//
//	template<typename TOP, typename TL, typename TR, template<typename > class F>
//	auto calculate(F<Expression<TOP, TL, TR>> const& f,
//			compact_index_type s) const
//					DECL_RET_TYPE(this->calculate(f.op_,
//									integer_sequence<size_t,field_traits<TL>::iform>(),f.lhs,f.rhs,s ) )
}
;

//template<typename G>
//template<typename T>
//typename field_traits<T>::value_type FiniteDiffMethod<G>::calculate(T const& f,
//		compact_index_type const &s) const
//{
//	CHECK(typeid(T).name());
//
//	return get_value(f, s);
//}

}
// namespace simpla

#endif /* FDM_H_ */
