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
template<typename, unsigned int> class Domain;

/** \ingroup DiffScheme
 *  \brief template of FvMesh
 */
template<typename G>
struct FiniteDiffMehtod
{

	typedef FiniteDiffMehtod<G> this_type;

	static constexpr unsigned int NUM_OF_COMPONENT_TYPE = G::NDIMS + 1;

	G const & geo;

	FiniteDiffMehtod(G const & g) :
			geo(g)
	{
	}
	FiniteDiffMehtod(this_type const & r) :
			geo(r.geo)
	{
	}
	~FiniteDiffMehtod() = default;

	this_type & operator=(this_type const &) = delete;

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************

private:

	template<typename TOP, typename TL>
	inline auto calculate_(TOP op, TL && f, typename G::index_type s) const
	DECL_RET_TYPE(op(get_value(std::forward<TL>(f),s) ) )

	template<typename TOP, typename TL, typename TR>
	inline auto calculate_(TOP op, TL && l, TR const &r,
			typename G::index_type s) const
			DECL_RET_TYPE(op(get_value(std::forward<TL>(l),s),get_value(r,s) ) )

	template<typename TM, typename TL>
	inline auto calculate_(_impl::ExteriorDerivative,
			_Field<Domain<TM, VERTEX>, TL> const & f,
			typename G::index_type s) const -> decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{
		auto D = geo.delta_index(s);

		return

		(get_value(f, s + D) * geo.volume(s + D)
				- get_value(f, s - D) * geo.volume(s - D)) * geo.inv_volume(s);
	}

	template<typename TM, typename TL>
	inline auto calculate_(_impl::ExteriorDerivative,
			_Field<Domain<TM, EDGE>, TL> const & f,
			typename G::index_type s) const -> decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{
		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return (

		(get_value(f, s + Y) * geo.volume(s + Y) //
		- get_value(f, s - Y) * geo.volume(s - Y)) //
		- (get_value(f, s + Z) * geo.volume(s + Z) //
		- get_value(f, s - Z) * geo.volume(s - Z)) //

		) * geo.inv_volume(s);
	}

	template<typename TM, typename TL>
	inline auto calculate_(_impl::ExteriorDerivative,
			_Field<Domain<TM, FACE>, TL> const & f,
			typename G::index_type s) const -> decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return (

		get_value(f, s + X) * geo.volume(s + X)

		- get_value(f, s - X) * geo.volume(s - X) //
		+ get_value(f, s + Y) * geo.volume(s + Y) //
		- get_value(f, s - Y) * geo.volume(s - Y) //
		+ get_value(f, s + Z) * geo.volume(s + Z) //
		- get_value(f, s - Z) * geo.volume(s - Z) //

		) * geo.inv_volume(s)

		;
	}

	template<typename TM, unsigned int IL, typename TL> void calculate_(
			_impl::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
			typename G::index_type s) const = delete;

	template<typename TM, unsigned int IL, typename TL> void calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, IL>, TL> const & f,
			typename G::index_type s) const = delete;

	template<typename TM, typename TL> inline auto calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, EDGE>, TL> const & f,
			typename G::index_type s) const ->decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		-(

		get_value(f, s + X) * geo.dual_volume(s + X)

		- get_value(f, s - X) * geo.dual_volume(s - X)

		+ get_value(f, s + Y) * geo.dual_volume(s + Y)

		- get_value(f, s - Y) * geo.dual_volume(s - Y)

		+ get_value(f, s + Z) * geo.dual_volume(s + Z)

		- get_value(f, s - Z) * geo.dual_volume(s - Z)

		) * geo.inv_dual_volume(s)

		;

	}

	template<typename TM, typename TL> inline auto calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, FACE>, TL> const & f,
			typename G::index_type s) const -> decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{
		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return

		-(

		(get_value(f, s + Y) * (geo.dual_volume(s + Y))
				- get_value(f, s - Y) * (geo.dual_volume(s - Y)))

				- (get_value(f, s + Z) * (geo.dual_volume(s + Z))
						- get_value(f, s - Z) * (geo.dual_volume(s - Z)))

		) * geo.inv_dual_volume(s)

		;
	}

	template<typename TM, typename TL> inline auto calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, VOLUME>, TL> const & f,
			typename G::index_type s) const -> decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{
		auto D = geo.delta_index(geo.dual(s));
		return

		-(

		get_value(f, s + D) * (geo.dual_volume(s + D)) //
		- get_value(f, s - D) * (geo.dual_volume(s - D))

		) * geo.inv_dual_volume(s)

		;
	}

//***************************************************************************************************

//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		return get_value(l, s) * get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, EDGE>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.delta_index(s);

		return (get_value(l, s - X) + get_value(l, s + X)) * 0.5
				* get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, FACE>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return (

		get_value(l, (s - Y) - Z) +

		get_value(l, (s - Y) + Z) +

		get_value(l, (s + Y) - Z) +

		get_value(l, (s + Y) + Z)

		) * 0.25 * get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
			_Field<Domain<TM, VOLUME>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return (

		get_value(l, ((s - X) - Y) - Z) +

		get_value(l, ((s - X) - Y) + Z) +

		get_value(l, ((s - X) + Y) - Z) +

		get_value(l, ((s - X) + Y) + Z) +

		get_value(l, ((s + X) - Y) - Z) +

		get_value(l, ((s + X) - Y) + Z) +

		get_value(l, ((s + X) + Y) - Z) +

		get_value(l, ((s + X) + Y) + Z)

		) * 0.125 * get_value(r, s);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.delta_index(s);
		return get_value(l, s) * (get_value(r, s - X) + get_value(r, s + X))
				* 0.5;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
			_Field<Domain<TM, EDGE>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = geo.delta_index(geo.roate(geo.dual(s)));
		auto Z = geo.delta_index(geo.inverse_roate(geo.dual(s)));

		return ((get_value(l, s - Y) + get_value(l, s + Y))
				* (get_value(l, s - Z) + get_value(l, s + Z)) * 0.25);
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
			_Field<Domain<TM, FACE>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

		(get_value(l, (s - Y) - Z) + get_value(l, (s - Y) + Z)
				+ get_value(l, (s + Y) - Z) + get_value(l, (s + Y) + Z))
				* (get_value(r, s - X) + get_value(r, s + X))
				+

				(get_value(l, (s - Z) - X) + get_value(l, (s - Z) + X)
						+ get_value(l, (s + Z) - X) + get_value(l, (s + Z) + X))
						* (get_value(r, s - Y) + get_value(r, s + Y))
				+

				(get_value(l, (s - X) - Y) + get_value(l, (s - X) + Y)
						+ get_value(l, (s + X) - Y) + get_value(l, (s + X) + Y))
						* (get_value(r, s - Z) + get_value(r, s + Z))

		) * 0.125;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, FACE>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = geo.delta_index(geo.roate(geo.dual(s)));
		auto Z = geo.delta_index(geo.inverse_roate(geo.dual(s)));

		return get_value(l, s)
				* (get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z)
						+ get_value(r, (s + Y) - Z) + get_value(r, (s + Y) + Z))
				* 0.25;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, FACE>, TL> const &r,
			_Field<Domain<TM, EDGE>, TR> const &l,
			typename G::index_type s) const->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

		(get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z)
				+ get_value(r, (s + Y) - Z) + get_value(r, (s + Y) + Z))
				* (get_value(l, s - X) + get_value(l, s + X))

				+ (get_value(r, (s - Z) - X) + get_value(r, (s - Z) + X)
						+ get_value(r, (s + Z) - X) + get_value(r, (s + Z) + X))
						* (get_value(l, s - Y) + get_value(l, s + Y))

				+ (get_value(r, (s - X) - Y) + get_value(r, (s - X) + Y)
						+ get_value(r, (s + X) - Y) + get_value(r, (s + X) + Y))
						* (get_value(l, s - Z) + get_value(l, s + Z))

		) * 0.125;
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::Wedge, _Field<Domain<TM, VOLUME>, TL> const &l,
			_Field<Domain<TM, VERTEX>, TR> const &r,
			typename G::index_type s) const->decltype(get_value(r,s)*get_value(l,s))
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		get_value(l, s) * (

		get_value(r, ((s - X) - Y) - Z) + //
				get_value(r, ((s - X) - Y) + Z) + //
				get_value(r, ((s - X) + Y) - Z) + //
				get_value(r, ((s - X) + Y) + Z) + //
				get_value(r, ((s + X) - Y) - Z) + //
				get_value(r, ((s + X) - Y) + Z) + //
				get_value(r, ((s + X) + Y) - Z) + //
				get_value(r, ((s + X) + Y) + Z) //

		) * 0.125;
	}

//***************************************************************************************************

	template<typename TM, unsigned int IL, typename TL> inline auto calculate_(
			_impl::HodgeStar, _Field<Domain<TM, IL>, TL> const & f,
			typename G::index_type s) const -> typename std::remove_reference<decltype(get_value(f,s))>::type
	{
//		auto X = geo.DI(0,s);
//		auto Y = geo.DI(1,s);
//		auto Z =geo.DI(2,s);
//
//		return
//
//		(
//
//		get_value(f,((s + X) - Y) - Z)*geo.inv_volume(((s + X) - Y) - Z) +
//
//		get_value(f,((s + X) - Y) + Z)*geo.inv_volume(((s + X) - Y) + Z) +
//
//		get_value(f,((s + X) + Y) - Z)*geo.inv_volume(((s + X) + Y) - Z) +
//
//		get_value(f,((s + X) + Y) + Z)*geo.inv_volume(((s + X) + Y) + Z) +
//
//		get_value(f,((s - X) - Y) - Z)*geo.inv_volume(((s - X) - Y) - Z) +
//
//		get_value(f,((s - X) - Y) + Z)*geo.inv_volume(((s - X) - Y) + Z) +
//
//		get_value(f,((s - X) + Y) - Z)*geo.inv_volume(((s - X) + Y) - Z) +
//
//		get_value(f,((s - X) + Y) + Z)*geo.inv_volume(((s - X) + Y) + Z)
//
//		) * 0.125 * geo.volume(s);

		return get_value(f, s) /** geo._impl::HodgeStarVolumeScale(s)*/;
	}

	template<typename TM, typename TL, typename TR> void calculate_(
			_impl::InteriorProduct, nTuple<G::NDIMS, TR> const & v,
			_Field<Domain<TM, VERTEX>, TL> const & f,
			typename G::index_type s) const = delete;

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::InteriorProduct, nTuple<G::NDIMS, TR> const & v,
			_Field<Domain<TM, EDGE>, TL> const & f,
			typename G::index_type s) const ->decltype(get_value(f,s)*v[0])
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(get_value(f, s + X) - get_value(f, s - X)) * 0.5 * v[0] //
		+ (get_value(f, s + Y) - get_value(f, s - Y)) * 0.5 * v[1] //
		+ (get_value(f, s + Z) - get_value(f, s - Z)) * 0.5 * v[2];
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::InteriorProduct, nTuple<G::NDIMS, TR> const & v,
			_Field<Domain<TM, FACE>, TL> const & f,
			typename G::index_type s) const ->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = geo.component_number(s);

		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);
		return

		(get_value(f, s + Y) + get_value(f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(get_value(f, s + Z) + get_value(f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TM, typename TL, typename TR> inline auto calculate_(
			_impl::InteriorProduct, nTuple<G::NDIMS, TR> const & v,
			_Field<Domain<TM, VOLUME>, TL> const & f,
			typename G::index_type s) const ->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = geo.component_number(geo.dual(s));
		unsigned int D = geo.delta_index(geo.dual(s));

		return (get_value(f, s + D) - get_value(f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<typename TM, unsigned int N, typename TL> inline auto calculate_(
			_impl::ExteriorDerivative, _Field<Domain<TM, EDGE>, TL> const & f,
			std::integral_constant<unsigned int, N>,
			typename G::index_type s) const -> decltype(get_value(f,s)-get_value(f,s))
	{

		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		Y = (geo.component_number(Y) == N) ? Y : 0UL;
		Z = (geo.component_number(Z) == N) ? Z : 0UL;

		return (get_value(f, s + Y) - get_value(f, s - Y))
				- (get_value(f, s + Z) - get_value(f, s - Z));
	}

	template<typename TM, unsigned int N, typename TL> inline auto calculate_(
			_impl::CodifferentialDerivative,
			_Field<Domain<TM, FACE>, TL> const & f,
			std::integral_constant<unsigned int, N>,
			typename G::index_type s) const -> decltype((get_value(f,s)-get_value(f,s))*std::declval<typename G::scalar_type>())
	{

		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		Y = (geo.component_number(Y) == N) ? Y : 0UL;
		Z = (geo.component_number(Z) == N) ? Z : 0UL;

		return (

		get_value(f, s + Y) * (geo.dual_volume(s + Y))      //
		- get_value(f, s - Y) * (geo.dual_volume(s - Y))    //
		- get_value(f, s + Z) * (geo.dual_volume(s + Z))    //
		+ get_value(f, s - Z) * (geo.dual_volume(s - Z))    //

		) * geo.inv_dual_volume(s);
	}
	template<typename TM, unsigned int IL, typename TR> inline auto calculate_(
			_impl::MapTo, std::integral_constant<unsigned int, IL> const &,
			_Field<Domain<TM, IL>, TR> const & f,
			typename G::index_type s) const
			DECL_RET_TYPE(get_value(f,s))

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, VERTEX> const &,
			_Field<Domain<TM, EDGE>, TR> const & f,
			typename G::index_type s) const ->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
	{

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<3,
				typename std::remove_reference<decltype(get_value(f,s))>::type>(
				{

				(get_value(f, s - X) + get_value(f, s + X)) * 0.5, //
				(get_value(f, s - Y) + get_value(f, s + Y)) * 0.5, //
				(get_value(f, s - Z) + get_value(f, s + Z)) * 0.5

				});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, EDGE> const &,
			_Field<Domain<TM, VERTEX>, TR> const & f,
			typename G::index_type s) const ->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = geo.component_number(s);
		auto D = geo.delta_index(s);

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, VERTEX> const &,
			_Field<Domain<TM, FACE>, TR> const & f,
			typename G::index_type s) const ->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
	{

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<3,
				typename std::remove_reference<decltype(get_value(f,s))>::type>(
				{ (

				get_value(f, (s - Y) - Z) +

				get_value(f, (s - Y) + Z) +

				get_value(f, (s + Y) - Z) +

				get_value(f, (s + Y) + Z)

				) * 0.25,

				(

				get_value(f, (s - Z) - X) +

				get_value(f, (s - Z) + X) +

				get_value(f, (s + Z) - X) +

				get_value(f, (s + Z) + X)

				) * 0.25,

				(

				get_value(f, (s - X) - Y) +

				get_value(f, (s - X) + Y) +

				get_value(f, (s + X) - Y) +

				get_value(f, (s + X) + Y)

				) * 0.25

				});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, FACE> const &,
			_Field<Domain<TM, VERTEX>, TR> const & f,
			typename G::index_type s) const ->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = geo.component_number(geo.dual(s));
		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return (

		(

		get_value(f, (s - Y) - Z)[n] +

		get_value(f, (s - Y) + Z)[n] +

		get_value(f, (s + Y) - Z)[n] +

		get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, VOLUME>,
			_Field<Domain<TM, FACE>, TR> const & f,
			typename G::index_type s) const ->nTuple<3,decltype(get_value(f,s) )>
	{

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<3,
				typename std::remove_reference<decltype(get_value(f,s))>::type>(
				{

				(get_value(f, s - X) + get_value(f, s + X)) * 0.5, //
				(get_value(f, s - Y) + get_value(f, s + Y)) * 0.5, //
				(get_value(f, s - Z) + get_value(f, s + Z)) * 0.5

				});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, FACE>,
			_Field<Domain<TM, VOLUME>, TR> const & f,
			typename G::index_type s) const ->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = geo.component_number(geo.dual(s));
		auto D = geo.delta_index(geo.dual(s));

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, VOLUME>,
			_Field<Domain<TM, EDGE>, TR> const & f,
			typename G::index_type s) const ->nTuple<3,typename std::remove_reference<decltype(get_value(f,s) )>::type>
	{

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<3,
				typename std::remove_reference<decltype(get_value(f,s))>::type>(
				{ (

				get_value(f, (s - Y) - Z) +

				get_value(f, (s - Y) + Z) +

				get_value(f, (s + Y) - Z) +

				get_value(f, (s + Y) + Z)

				) * 0.25,

				(

				get_value(f, (s - Z) - X) +

				get_value(f, (s - Z) + X) +

				get_value(f, (s + Z) - X) +

				get_value(f, (s + Z) + X)

				) * 0.25,

				(

				get_value(f, (s - X) - Y) +

				get_value(f, (s - X) + Y) +

				get_value(f, (s + X) - Y) +

				get_value(f, (s + X) + Y)

				) * 0.25,

				});
	}

	template<typename TM, typename TR> inline auto calculate_(_impl::MapTo,
			std::integral_constant<unsigned int, EDGE>,
			_Field<Domain<TM, VOLUME>, TR> const & f,
			typename G::index_type s) const ->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = geo.component_number(geo.dual(s));
		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);
		return (

		(

		get_value(f, (s - Y) - Z)[n] +

		get_value(f, (s - Y) + Z)[n] +

		get_value(f, (s + Y) - Z)[n] +

		get_value(f, (s + Y) + Z)[n]

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
