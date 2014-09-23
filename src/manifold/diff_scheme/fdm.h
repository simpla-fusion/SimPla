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

#include "../utilities/sp_type_traits.h"
#include "../utilities/container_dense.h"
#include "../physics/constants.h"

namespace simpla
{

template<typename, typename > class Field;
template<typename, unsigned int> class Domain;
class HodgeStar;
class InteriorProduct;
class Wedge;
class ExteriorDerivative;
class CodifferentialDerivative;
class MapTo;

/** \ingroup DiffScheme
 *  \brief template of FvMesh
 */
template<typename TGeometry>
struct FiniteDiffMehtod
{
	typedef TGeometry manifold_type;

	typedef FiniteDiffMehtod<manifold_type> this_type;

	typedef typename manifold_type::topology_type topology_type;

	typedef typename manifold_type::coordinates_type coordinates_type;

	typedef typename manifold_type::index_type index_type;

	typedef Real scalar_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr unsigned int NUM_OF_COMPONENT_TYPE = NDIMS + 1;

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************
	template<typename TOP, typename TL> static inline auto eval(TOP const & op,
			manifold_type const & geo, TL const & f, index_type s) const
			DECL_RET_TYPE(op(get_value(f,s) ) )

	template<typename TOP, typename TL, typename TR> static inline auto eval(
			TOP const & op, manifold_type const & geo, TL const & l,
			TR const &r, index_type s) const
			DECL_RET_TYPE(op(get_value(l,s),get_value(r,s) ) )

	template<typename TL> static inline auto eval(ExteriorDerivative,
			manifold_type const & geo,
			Field<Domain<manifold_type, VERTEX>, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = geo.delta_index(s);

		return

		(get_value(f, s + D) * geo.volume(s + D)
				- get_value(f, s - D) * geo.volume(s - D)) * geo.inv_volume(s);
	}

	template<typename TL> static inline auto eval(ExteriorDerivative,
			manifold_type const & geo,
			Field<Domain<manifold_type, EDGE>, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
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

	template<typename TL> static inline auto eval(ExteriorDerivative,
			manifold_type const & geo,
			Field<Domain<manifold_type, FACE>, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
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

	template<unsigned int IL, typename TL> void eval(ExteriorDerivative,
			manifold_type const & geo, Field<this_type, IL, TL> const & f,
			index_type s) const = delete;

	template<unsigned int IL, typename TL> void eval(CodifferentialDerivative,
			manifold_type const & geo, Field<this_type, IL, TL> const & f,
			index_type s) const = delete;

	template<typename TL> static inline auto eval(CodifferentialDerivative,
			manifold_type const & geo,
			Field<Domain<manifold_type, EDGE>, TL> const & f,
			index_type s) const->decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
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

	template<typename TL> static inline auto eval(CodifferentialDerivative,
			manifold_type const & geo,
			Field<Domain<manifold_type, FACE>, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
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

	template<typename TL> static inline auto eval(CodifferentialDerivative,
			manifold_type const & geo,
			Field<Domain<manifold_type, VOLUME>, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
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
	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, VERTEX>, TL> const &l,
			Field<Domain<manifold_type, VERTEX>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		return get_value(l, s) * get_value(r, s);
	}

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, VERTEX>, TL> const &l,
			Field<Domain<manifold_type, EDGE>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.delta_index(s);

		return (get_value(l, s - X) + get_value(l, s + X)) * 0.5
				* get_value(r, s);
	}

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, VERTEX>, TL> const &l,
			Field<Domain<manifold_type, FACE>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, VERTEX>, TL> const &l,
			Field<Domain<manifold_type, VOLUME>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, EDGE>, TL> const &l,
			Field<Domain<manifold_type, VERTEX>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = geo.delta_index(s);
		return get_value(l, s) * (get_value(r, s - X) + get_value(r, s + X))
				* 0.5;
	}

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, EDGE>, TL> const &l,
			Field<Domain<manifold_type, EDGE>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = geo.delta_index(geo.roate(geo.dual(s)));
		auto Z = geo.delta_index(geo.inverse_roate(geo.dual(s)));

		return ((get_value(l, s - Y) + get_value(l, s + Y))
				* (get_value(l, s - Z) + get_value(l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, EDGE>, TL> const &l,
			Field<Domain<manifold_type, FACE>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, FACE>, TL> const &l,
			Field<Domain<manifold_type, VERTEX>, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = geo.delta_index(geo.roate(geo.dual(s)));
		auto Z = geo.delta_index(geo.inverse_roate(geo.dual(s)));

		return get_value(l, s)
				* (get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z)
						+ get_value(r, (s + Y) - Z) + get_value(r, (s + Y) + Z))
				* 0.25;
	}

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, FACE>, TL> const &r,
			Field<Domain<manifold_type, EDGE>, TR> const &l,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
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

	template<typename TL, typename TR> static inline auto eval(Wedge,
			manifold_type const & geo,
			Field<Domain<manifold_type, VOLUME>, TL> const &l,
			Field<Domain<manifold_type, VERTEX>, TR> const &r,
			index_type s) const ->decltype(get_value(r,s)*get_value(l,s))
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

	template<unsigned int IL, typename TL> static inline auto eval(HodgeStar,
			manifold_type const & geo, Field<this_type, IL, TL> const & f,
			index_type s) const-> typename std::remove_reference<decltype(get_value(f,s))>::type
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

		return get_value(f, s) /** geo.HodgeStarVolumeScale(s)*/;
	}

	template<typename TL, typename TR> void eval(InteriorProduct,
			manifold_type const & geo, nTuple<NDIMS, TR> const & v,
			Field<Domain<manifold_type, VERTEX>, TL> const & f,
			index_type s) const = delete;

	template<typename TL, typename TR> static inline auto eval(InteriorProduct,
			manifold_type const & geo, nTuple<NDIMS, TR> const & v,
			Field<Domain<manifold_type, EDGE>, TL> const & f,
			index_type s) const->decltype(get_value(f,s)*v[0])
	{
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(get_value(f, s + X) - get_value(f, s - X)) * 0.5 * v[0] //
		+ (get_value(f, s + Y) - get_value(f, s - Y)) * 0.5 * v[1] //
		+ (get_value(f, s + Z) - get_value(f, s - Z)) * 0.5 * v[2];
	}

	template<typename TL, typename TR> static inline auto eval(InteriorProduct,
			manifold_type const & geo, nTuple<NDIMS, TR> const & v,
			Field<Domain<manifold_type, FACE>, TL> const & f,
			index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = geo.component_number(s);

		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);
		return

		(get_value(f, s + Y) + get_value(f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(get_value(f, s + Z) + get_value(f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL, typename TR> static inline auto eval(InteriorProduct,
			manifold_type const & geo, nTuple<NDIMS, TR> const & v,
			Field<Domain<manifold_type, VOLUME>, TL> const & f,
			index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = geo.component_number(geo.dual(s));
		unsigned int D = geo.delta_index(geo.dual(s));

		return (get_value(f, s + D) - get_value(f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<unsigned int N, typename TL> static inline auto eval(
			ExteriorDerivative, manifold_type const & geo,
			Field<Domain<manifold_type, EDGE>, TL> const & f,
			std::integral_constant<unsigned int, N>,
			index_type s) const-> decltype(get_value(f,s)-get_value(f,s))
	{

		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		Y = (geo.component_number(Y) == N) ? Y : 0UL;
		Z = (geo.component_number(Z) == N) ? Z : 0UL;

		return (get_value(f, s + Y) - get_value(f, s - Y))
				- (get_value(f, s + Z) - get_value(f, s - Z));
	}

	template<unsigned int N, typename TL> static inline auto eval(
			CodifferentialDerivative, manifold_type const & geo,
			Field<Domain<manifold_type, FACE>, TL> const & f,
			std::integral_constant<unsigned int, N>,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
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
	template<unsigned int IL, typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, IL> const &,
			Field<this_type, IL, TR> const & f, index_type s) const
			DECL_RET_TYPE(get_value(f,s))

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, VERTEX> const &,
			Field<Domain<manifold_type, EDGE>, TR> const & f,
			index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
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

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, EDGE> const &,
			Field<Domain<manifold_type, VERTEX>, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = geo.component_number(s);
		auto D = geo.delta_index(s);

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, VERTEX> const &,
			Field<Domain<manifold_type, FACE>, TR> const & f,
			index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
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

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, FACE> const &,
			Field<Domain<manifold_type, VERTEX>, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
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

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, VOLUME>,
			Field<Domain<manifold_type, FACE>, TR> const & f,
			index_type s) const->nTuple<3,decltype(get_value(f,s) )>
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

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, FACE>,
			Field<Domain<manifold_type, VOLUME>, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = geo.component_number(geo.dual(s));
		auto D = geo.delta_index(geo.dual(s));

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, VOLUME>,
			Field<Domain<manifold_type, EDGE>, TR> const & f,
			index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s) )>::type>
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

	template<typename TR> static inline auto eval(MapTo,
			manifold_type const & geo,
			std::integral_constant<unsigned int, EDGE>,
			Field<Domain<manifold_type, VOLUME>, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
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
};

}
// namespace simpla

#endif /* FDM_H_ */
