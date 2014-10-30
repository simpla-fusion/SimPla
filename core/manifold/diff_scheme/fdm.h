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

	static constexpr size_t NUM_OF_COMPONENT_TYPE = G::ndims + 1;

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


	template<typename T>
	auto _get_value(T const & v, typename G::compact_index_type s) const
	DECL_RET_TYPE((get_value(v,s)))

	template<typename TOP, typename TL, template<typename > class _F>
	auto _get_value(_F<Expression<TOP, TL> > const & f,
			typename G::compact_index_type s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs_,s)))

	template<typename TOP, typename TL, typename TR,
			template<typename > class _F>
	auto _get_value(_F<Expression<TOP, TL, TR> > const & f,
			typename G::compact_index_type s) const
			DECL_RET_TYPE((calculate_(f.op_,f.lhs_,f.rhs_,s)))

	template<typename TOP, typename TL>
	inline auto calculate_(TOP op, TL && f,
			typename G::compact_index_type s) const
			DECL_RET_TYPE(op(_get_value(std::forward<TL>(f),s) ) )

	template<typename TOP, typename TL, typename TR>
	inline auto calculate_(TOP op, TL && l, TR const &r,
			typename G::compact_index_type s) const
					DECL_RET_TYPE(op(_get_value(std::forward<TL>(l),s),_get_value(r,s) ) )

	template<typename ... TL>
	inline auto calculate_(_impl::ExteriorDerivative, _Field<TL...> const & f,
			typename G::compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==VERTEX,
			decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())>::type
	{
		auto D = geo->delta_index(s);
		return (_get_value(f, s + D) * geo->volume(s + D)
				- _get_value(f, s - D) * geo->volume(s - D))
				* geo->inv_volume(s);
	}

	template<typename ...TL>
	inline auto calculate_(_impl::ExteriorDerivative, _Field<TL...> const & f,
			typename G::compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==EDGE,
			decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())
			>::type
	{
		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return (

		(

		_get_value(f, s + Y) * geo->volume(s + Y) //
		- _get_value(f, s - Y) * geo->volume(s - Y) //

		) - (

		_get_value(f, s + Z) * geo->volume(s + Z) //
		- _get_value(f, s - Z) * geo->volume(s - Z) //

		)

		) * geo->inv_volume(s);

	}

	template<typename ... TL>
	inline auto calculate_(_impl::ExteriorDerivative, _Field<TL...> const & f,
			typename G::compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==FACE,
			decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())
			>::type
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return (

		_get_value(f, s + X) * geo->volume(s + X)

		- _get_value(f, s - X) * geo->volume(s - X) //
		+ _get_value(f, s + Y) * geo->volume(s + Y) //
		- _get_value(f, s - Y) * geo->volume(s - Y) //
		+ _get_value(f, s + Z) * geo->volume(s + Z) //
		- _get_value(f, s - Z) * geo->volume(s - Z) //

		) * geo->inv_volume(s)

		;
	}

	template<typename TM, size_t IL, typename TL> void calculate_(
			_impl::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
			typename G::compact_index_type s) const = delete;

	template<typename TM, size_t IL, typename TL> void calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, IL>, TL> const & f,
			typename G::compact_index_type s) const = delete;

	template<typename ... TL> inline auto calculate_(
			_impl::CodifferentialDerivative, _Field<TL...> const & f,
			typename G::compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==EDGE,
			decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())>::type
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		-(

		_get_value(f, s + X) * geo->dual_volume(s + X)

		- _get_value(f, s - X) * geo->dual_volume(s - X)

		+ _get_value(f, s + Y) * geo->dual_volume(s + Y)

		- _get_value(f, s - Y) * geo->dual_volume(s - Y)

		+ _get_value(f, s + Z) * geo->dual_volume(s + Z)

		- _get_value(f, s - Z) * geo->dual_volume(s - Z)

		) * geo->inv_dual_volume(s)

		;

	}

	template<typename ...TL>
	inline auto calculate_(_impl::CodifferentialDerivative,
			_Field<TL...> const & f,
			typename G::compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==FACE,
			decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())>::type
	{
		auto X = geo->delta_index(s);
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return

		-(

		(_get_value(f, s + Y) * (geo->dual_volume(s + Y))
				- _get_value(f, s - Y) * (geo->dual_volume(s - Y)))

				- (_get_value(f, s + Z) * (geo->dual_volume(s + Z))
						- _get_value(f, s - Z) * (geo->dual_volume(s - Z)))

		) * geo->inv_dual_volume(s)

		;
	}

	template<typename ...TL> inline auto calculate_(
			_impl::CodifferentialDerivative, _Field<TL...> const & f,
			typename G::compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==VOLUME,
			decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())>::type
	{
		auto D = geo->delta_index(geo->dual(s));
		return

		-(

		_get_value(f, s + D) * (geo->dual_volume(s + D)) //
		- _get_value(f, s - D) * (geo->dual_volume(s - D))

		) * geo->inv_dual_volume(s)

		;
	}

//***************************************************************************************************

//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		return _get_value(l, s) * _get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, EDGE>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto X = geo->delta_index(s);

		return (_get_value(l, s - X) + _get_value(l, s + X)) * 0.5
				* _get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, FACE>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return (

		_get_value(l, (s - Y) - Z) +

		_get_value(l, (s - Y) + Z) +

		_get_value(l, (s + Y) - Z) +

		_get_value(l, (s + Y) + Z)

		) * 0.25 * _get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, VOLUME>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return (

		_get_value(l, ((s - X) - Y) - Z) +

		_get_value(l, ((s - X) - Y) + Z) +

		_get_value(l, ((s - X) + Y) - Z) +

		_get_value(l, ((s - X) + Y) + Z) +

		_get_value(l, ((s + X) - Y) - Z) +

		_get_value(l, ((s + X) - Y) + Z) +

		_get_value(l, ((s + X) + Y) - Z) +

		_get_value(l, ((s + X) + Y) + Z)

		) * 0.125 * _get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto X = geo->delta_index(s);
		return _get_value(l, s) * (_get_value(r, s - X) + _get_value(r, s + X))
				* 0.5;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
			_Field<Domain<TM, EDGE>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto Y = geo->delta_index(geo->roate(geo->dual(s)));
		auto Z = geo->delta_index(geo->inverse_roate(geo->dual(s)));

		return ((_get_value(l, s - Y) + _get_value(l, s + Y))
				* (_get_value(l, s - Z) + _get_value(l, s + Z)) * 0.25);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
			_Field<Domain<TM, FACE>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		(

		(_get_value(l, (s - Y) - Z) + _get_value(l, (s - Y) + Z)
				+ _get_value(l, (s + Y) - Z) + _get_value(l, (s + Y) + Z))
				* (_get_value(r, s - X) + _get_value(r, s + X))
				+

				(_get_value(l, (s - Z) - X) + _get_value(l, (s - Z) + X)
						+ _get_value(l, (s + Z) - X)
						+ _get_value(l, (s + Z) + X))
						* (_get_value(r, s - Y) + _get_value(r, s + Y))
				+

				(_get_value(l, (s - X) - Y) + _get_value(l, (s - X) + Y)
						+ _get_value(l, (s + X) - Y)
						+ _get_value(l, (s + X) + Y))
						* (_get_value(r, s - Z) + _get_value(r, s + Z))

		) * 0.125;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, FACE>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto Y = geo->delta_index(geo->roate(geo->dual(s)));
		auto Z = geo->delta_index(geo->inverse_roate(geo->dual(s)));

		return _get_value(l, s)
				* (_get_value(r, (s - Y) - Z) + _get_value(r, (s - Y) + Z)
						+ _get_value(r, (s + Y) - Z)
						+ _get_value(r, (s + Y) + Z)) * 0.25;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, FACE>, TL> const &r,
			_Field<Domain<TM, EDGE>, TR> const &l,
			typename G::compact_index_type s) const->decltype(_get_value(l,s)*_get_value(r,s))
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		(

		(_get_value(r, (s - Y) - Z) + _get_value(r, (s - Y) + Z)
				+ _get_value(r, (s + Y) - Z) + _get_value(r, (s + Y) + Z))
				* (_get_value(l, s - X) + _get_value(l, s + X))

				+ (_get_value(r, (s - Z) - X) + _get_value(r, (s - Z) + X)
						+ _get_value(r, (s + Z) - X)
						+ _get_value(r, (s + Z) + X))
						* (_get_value(l, s - Y) + _get_value(l, s + Y))

				+ (_get_value(r, (s - X) - Y) + _get_value(r, (s - X) + Y)
						+ _get_value(r, (s + X) - Y)
						+ _get_value(r, (s + X) + Y))
						* (_get_value(l, s - Z) + _get_value(l, s + Z))

		) * 0.125;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VOLUME>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::compact_index_type s) const->decltype(_get_value(r,s)*_get_value(l,s))
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		_get_value(l, s) * (

		_get_value(r, ((s - X) - Y) - Z) + //
				_get_value(r, ((s - X) - Y) + Z) + //
				_get_value(r, ((s - X) + Y) - Z) + //
				_get_value(r, ((s - X) + Y) + Z) + //
				_get_value(r, ((s + X) - Y) - Z) + //
				_get_value(r, ((s + X) - Y) + Z) + //
				_get_value(r, ((s + X) + Y) - Z) + //
				_get_value(r, ((s + X) + Y) + Z) //

		) * 0.125;
	}

//***************************************************************************************************

	template<typename TM, size_t IL, typename TL> inline auto calculate_(
			_impl::HodgeStar, _Field<Domain<TM, IL>, TL> const & f,
			typename G::compact_index_type s) const -> typename std::remove_reference<decltype(_get_value(f,s))>::type
	{
//		auto X = geo->DI(0,s);
//		auto Y = geo->DI(1,s);
//		auto Z =geo->DI(2,s);
//
//		return
//
//		(
//
//		_get_value(f,((s + X) - Y) - Z)*geo->inv_volume(((s + X) - Y) - Z) +
//
//		_get_value(f,((s + X) - Y) + Z)*geo->inv_volume(((s + X) - Y) + Z) +
//
//		_get_value(f,((s + X) + Y) - Z)*geo->inv_volume(((s + X) + Y) - Z) +
//
//		_get_value(f,((s + X) + Y) + Z)*geo->inv_volume(((s + X) + Y) + Z) +
//
//		_get_value(f,((s - X) - Y) - Z)*geo->inv_volume(((s - X) - Y) - Z) +
//
//		_get_value(f,((s - X) - Y) + Z)*geo->inv_volume(((s - X) - Y) + Z) +
//
//		_get_value(f,((s - X) + Y) - Z)*geo->inv_volume(((s - X) + Y) - Z) +
//
//		_get_value(f,((s - X) + Y) + Z)*geo->inv_volume(((s - X) + Y) + Z)
//
//		) * 0.125 * geo->volume(s);

		return _get_value(f, s) /** geo->_impl::HodgeStarVolumeScale(s)*/;
	}

	template<typename TM, typename TL, typename TR> void calculate_(
			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
			_Field<Domain<TM, VERTEX>, TL> const & f,
			typename G::compact_index_type s) const = delete;

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
			_Field<Domain<TM, EDGE>, TL> const & f,
			typename G::compact_index_type s) const ->decltype(_get_value(f,s)*v[0])
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		(_get_value(f, s + X) - _get_value(f, s - X)) * 0.5 * v[0] //
		+ (_get_value(f, s + Y) - _get_value(f, s - Y)) * 0.5 * v[1] //
		+ (_get_value(f, s + Z) - _get_value(f, s - Z)) * 0.5 * v[2];
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
			_Field<Domain<TM, FACE>, TL> const & f,
			typename G::compact_index_type s) const ->decltype(_get_value(f,s)*v[0])
	{
		size_t n = geo->component_number(s);

		auto X = geo->delta_index(s);
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);
		return

		(_get_value(f, s + Y) + _get_value(f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(_get_value(f, s + Z) + _get_value(f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
			_Field<Domain<TM, VOLUME>, TL> const & f,
			typename G::compact_index_type s) const ->decltype(_get_value(f,s)*v[0])
	{
		size_t n = geo->component_number(geo->dual(s));
		size_t D = geo->delta_index(geo->dual(s));

		return (_get_value(f, s + D) - _get_value(f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<typename TM, size_t N, typename TL> inline auto calculate_(
			_impl::ExteriorDerivative, _Field<Domain<TM, EDGE>, TL> const & f,
			std::integral_constant<size_t, N>,
			typename G::compact_index_type s) const -> decltype(_get_value(f,s)-_get_value(f,s))
	{

		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		Y = (geo->component_number(Y) == N) ? Y : 0UL;
		Z = (geo->component_number(Z) == N) ? Z : 0UL;

		return (_get_value(f, s + Y) - _get_value(f, s - Y))
				- (_get_value(f, s + Z) - _get_value(f, s - Z));
	}

	template<typename TM, size_t N, typename TL> inline auto calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, FACE>, TL> const & f,
			std::integral_constant<size_t, N>,
			typename G::compact_index_type s) const -> decltype((_get_value(f,s)-_get_value(f,s))*std::declval<typename G::scalar_type>())
	{

		auto X = geo->delta_index(s);
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		Y = (geo->component_number(Y) == N) ? Y : 0UL;
		Z = (geo->component_number(Z) == N) ? Z : 0UL;

		return (

		_get_value(f, s + Y) * (geo->dual_volume(s + Y))      //
		- _get_value(f, s - Y) * (geo->dual_volume(s - Y))    //
		- _get_value(f, s + Z) * (geo->dual_volume(s + Z))    //
		+ _get_value(f, s - Z) * (geo->dual_volume(s - Z))    //

		) * geo->inv_dual_volume(s);
	}
	template<typename TM, size_t IL, typename TR> inline auto calculate_(
			_impl::MapTo, std::integral_constant<size_t, IL> const &,
			_Field<Domain<TM, IL>, TR> const & f,
			typename G::compact_index_type s) const
			DECL_RET_TYPE(_get_value(f,s))

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, VERTEX> const &,
			_Field<Domain<TM, EDGE>, TR> const & f,
			typename G::compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(_get_value(f,s))>::type,3>
	{

		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return nTuple<
				typename std::remove_reference<decltype(_get_value(f,s))>::type,
				3>(
		{

		(_get_value(f, s - X) + _get_value(f, s + X)) * 0.5, //
		(_get_value(f, s - Y) + _get_value(f, s + Y)) * 0.5, //
		(_get_value(f, s - Z) + _get_value(f, s + Z)) * 0.5

		});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, EDGE> const &,
			_Field<Domain<TM, VERTEX>, TR> const & f,
			typename G::compact_index_type s) const ->typename std::remove_reference<decltype(_get_value(f,s)[0])>::type
	{

		auto n = geo->component_number(s);
		auto D = geo->delta_index(s);

		return ((_get_value(f, s - D)[n] + _get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, VERTEX> const &,
			_Field<Domain<TM, FACE>, TR> const & f,
			typename G::compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(_get_value(f,s))>::type,3>
	{

		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return nTuple<
				typename std::remove_reference<decltype(_get_value(f,s))>::type,
				3>(
		{ (

		_get_value(f, (s - Y) - Z) +

		_get_value(f, (s - Y) + Z) +

		_get_value(f, (s + Y) - Z) +

		_get_value(f, (s + Y) + Z)

		) * 0.25,

		(

		_get_value(f, (s - Z) - X) +

		_get_value(f, (s - Z) + X) +

		_get_value(f, (s + Z) - X) +

		_get_value(f, (s + Z) + X)

		) * 0.25,

		(

		_get_value(f, (s - X) - Y) +

		_get_value(f, (s - X) + Y) +

		_get_value(f, (s + X) - Y) +

		_get_value(f, (s + X) + Y)

		) * 0.25

		});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, FACE> const &,
			_Field<Domain<TM, VERTEX>, TR> const & f,
			typename G::compact_index_type s) const ->typename std::remove_reference<decltype(_get_value(f,s)[0])>::type
	{

		auto n = geo->component_number(geo->dual(s));
		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return (

		(

		_get_value(f, (s - Y) - Z)[n] +

		_get_value(f, (s - Y) + Z)[n] +

		_get_value(f, (s + Y) - Z)[n] +

		_get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, VOLUME>,
			_Field<Domain<TM, FACE>, TR> const & f,
			typename G::compact_index_type s) const ->nTuple<decltype(_get_value(f,s) ),3>
	{

		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return nTuple<
				typename std::remove_reference<decltype(_get_value(f,s))>::type,
				3>(
		{

		(_get_value(f, s - X) + _get_value(f, s + X)) * 0.5, //
		(_get_value(f, s - Y) + _get_value(f, s + Y)) * 0.5, //
		(_get_value(f, s - Z) + _get_value(f, s + Z)) * 0.5

		});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, FACE>,
			_Field<Domain<TM, VOLUME>, TR> const & f,
			typename G::compact_index_type s) const ->typename std::remove_reference<decltype(_get_value(f,s)[0])>::type
	{

		auto n = geo->component_number(geo->dual(s));
		auto D = geo->delta_index(geo->dual(s));

		return ((_get_value(f, s - D)[n] + _get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, VOLUME>,
			_Field<Domain<TM, EDGE>, TR> const & f,
			typename G::compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(_get_value(f,s) )>::type,3>
	{

		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return nTuple<
				typename std::remove_reference<decltype(_get_value(f,s))>::type,
				3>(
		{ (

		_get_value(f, (s - Y) - Z) +

		_get_value(f, (s - Y) + Z) +

		_get_value(f, (s + Y) - Z) +

		_get_value(f, (s + Y) + Z)

		) * 0.25,

		(

		_get_value(f, (s - Z) - X) +

		_get_value(f, (s - Z) + X) +

		_get_value(f, (s + Z) - X) +

		_get_value(f, (s + Z) + X)

		) * 0.25,

		(

		_get_value(f, (s - X) - Y) +

		_get_value(f, (s - X) + Y) +

		_get_value(f, (s + X) - Y) +

		_get_value(f, (s + X) + Y)

		) * 0.25,

		});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<size_t, EDGE>,
			_Field<TR, Domain<TM, VOLUME>> const & f,
			typename G::compact_index_type s) const ->typename std::remove_reference<decltype(_get_value(f,s)[0])>::type
	{

		auto n = geo->component_number(geo->dual(s));
		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);
		return (

		(

		_get_value(f, (s - Y) - Z)[n] +

		_get_value(f, (s - Y) + Z)[n] +

		_get_value(f, (s + Y) - Z)[n] +

		_get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}
public:
	template<typename ...Args>
	auto calculate(Args && ... args) const
	DECL_RET_TYPE( (this->calculate_( std::forward<Args>(args)...)) )

};

}
// namespace simpla

#endif /* FDM_H_ */
