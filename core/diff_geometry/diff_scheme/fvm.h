/*
 * mesh_rectangle.h
 *
 *  created on: 2014-2-26
 *      Author: salmon
 */

#ifndef MESH_RECTANGLE_H_
#define MESH_RECTANGLE_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <type_traits>

#include "../../data_structure/container_dense.h"
#include "../utilities/sp_type_traits.h"
#include "../physics/constants.h"

namespace simpla
{

template<typename ...> class _Field;

/** @ingroup diff_scheme
 *  \brief   FvMesh
 */
template<typename TGeometry>
class FiniteVolumeMethod
{
public:
	typedef TGeometry geometry_type;

	typedef FiniteVolumeMethod<geometry_type> this_type;

	typedef typename geo.topology_type topology_type;

	typedef typename geo.coordinates_type coordinates_type;

	typedef typename geo.index_type index_type;

	typedef typename geo.index_type index_type;

//	typedef Interpolator<this_type, std::nullptr_t> interpolator_type;

	static constexpr unsigned int NDIMS = 3;

	static constexpr unsigned int NUM_OF_COMPONENT_TYPE = NDIMS + 1;

	FiniteVolumeMethod()
	{
	}

	~FiniteVolumeMethod()
	{
	}

	FiniteVolumeMethod(const this_type&) = delete;

	template<typename TDict>
	bool load(TDict const& dict)
	{
		return geo.load(dict);
	}

//***************************************************************************************************
// Exterior algebra
//***************************************************************************************************

	template<typename TL> inline auto calculus(ExteriorDerivative,
			geometry_type const & geo, _Field<this_type, VERTEX, TL> const & f,
			index_type s) const-> decltype((geo.get_value(f,s)-geo.get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = geo.topology_type::delta_index(s);

		return

		(geo.get_value(f, s + D) * geo.volume(s + D)
				- get_value(f, s - D) * geo.volume(s - D)) * geo.inv_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
				+ get_value(f, s + D) * k_imag[geo.component_number(D)];
#endif
		;
	}

	template<typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
			_Field<this_type, EDGE, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

		(get_value(f, s + Y) * geo.volume(s + Y) //
		- get_value(f, s - Y) * geo.volume(s - Y)) //
		- (get_value(f, s + Z) * geo.volume(s + Z) //
		- get_value(f, s - Z) * geo.volume(s - Z)) //

		) * geo.inv_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
				+ get_value(f, s + Y) * k_imag[geo.component_number(Y)]
				- get_value(f, s + Z) * k_imag[geo.component_number(Z)]
#endif

		;
	}

	template<typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
			_Field<this_type, FACE, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return (

		get_value(f, s + X) * geo.volume(s + X)

		- get_value(f, s - X) * geo.volume(s - X) //
		+ get_value(f, s + Y) * geo.volume(s + Y) //
		- get_value(f, s - Y) * geo.volume(s - Y) //
		+ get_value(f, s + Z) * geo.volume(s + Z) //
		- get_value(f, s - Z) * geo.volume(s - Z) //

		) * geo.inv_volume(s)

#ifndef DISABLE_SPECTRAL_METHD

				+ get_value(f, s + X) * k_imag[geo.component_number(X)]
				+ get_value(f, s + Y) * k_imag[geo.component_number(Y)]
				+ get_value(f, s + Z) * k_imag[geo.component_number(Z)]

#endif

		;
	}

	template<unsigned int IL, typename TL> void Opcalculus(
			std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
			_Field<this_type, IL, TL> const & f, index_type s) const = delete;

	template<unsigned int IL, typename TL> void Opcalculus(
			std::integral_constant<unsigned int, CODIFFERENTIAL>,
			_Field<this_type, IL, TL> const & f, index_type s) const = delete;

	template<typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, CODIFFERENTIAL>,
			_Field<this_type, EDGE, TL> const & f,
			index_type s) const->decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		-(

		get_value(f, s + X) * geo.dual_volume(s + X)

		- get_value(f, s - X) * geo.dual_volume(s - X)

		+ get_value(f, s + Y) * geo.dual_volume(s + Y)

		- get_value(f, s - Y) * geo.dual_volume(s - Y)

		+ get_value(f, s + Z) * geo.dual_volume(s + Z)

		- get_value(f, s - Z) * geo.dual_volume(s - Z)

		) * geo.inv_dual_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
				- get_value(f, s + X) * k_imag[geo.component_number(X)]
				- get_value(f, s + Y) * k_imag[geo.component_number(Y)]
				- get_value(f, s + Z) * k_imag[geo.component_number(Z)]
#endif
		;

	}

	template<typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, CODIFFERENTIAL>,
			_Field<this_type, FACE, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return

		-(

		(get_value(f, s + Y) * (geo.dual_volume(s + Y))
				- get_value(f, s - Y) * (geo.dual_volume(s - Y)))

				- (get_value(f, s + Z) * (geo.dual_volume(s + Z))
						- get_value(f, s - Z) * (geo.dual_volume(s - Z)))

		) * geo.inv_dual_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
				- get_value(f, s + Y) * k_imag[geo.component_number(Y)]
				+ get_value(f, s + Z) * k_imag[geo.component_number(Z)]
#endif
		;
	}

	template<typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, CODIFFERENTIAL>,
			_Field<this_type, VOLUME, TL> const & f,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{
		auto D = topology_type::delta_index(topology_type::dual(s));
		return

		-(

		get_value(f, s + D) * (geo.dual_volume(s + D)) //
		- get_value(f, s - D) * (geo.dual_volume(s - D))

		) * geo.inv_dual_volume(s)

#ifndef DISABLE_SPECTRAL_METHD
				- get_value(f, s + D) * k_imag[geo.component_number(D)]
#endif

		;
	}

//***************************************************************************************************

//! Form<IR> ^ Form<IR> => Form<IR+IL>
	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, VERTEX, TL> const &l,
			_Field<this_type, VERTEX, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		return get_value(l, s) * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, VERTEX, TL> const &l,
			_Field<this_type, EDGE, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::delta_index(s);

		return (get_value(l, s - X) + get_value(l, s + X)) * 0.5
				* get_value(r, s);
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, VERTEX, TL> const &l,
			_Field<this_type, FACE, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

		get_value(l, (s - Y) - Z) +

		get_value(l, (s - Y) + Z) +

		get_value(l, (s + Y) - Z) +

		get_value(l, (s + Y) + Z)

		) * 0.25 * get_value(r, s);
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, VERTEX, TL> const &l,
			_Field<this_type, VOLUME, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

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

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, EDGE, TL> const &l,
			_Field<this_type, VERTEX, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::delta_index(s);
		return get_value(l, s) * (get_value(r, s - X) + get_value(r, s + X))
				* 0.5;
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, EDGE, TL> const &l,
			_Field<this_type, EDGE, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = topology_type::delta_index(
				topology_type::roate(topology_type::dual(s)));
		auto Z = topology_type::delta_index(
				topology_type::inverse_roate(topology_type::dual(s)));

		return ((get_value(l, s - Y) + get_value(l, s + Y))
				* (get_value(l, s - Z) + get_value(l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, EDGE, TL> const &l,
			_Field<this_type, FACE, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

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

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, FACE, TL> const &l,
			_Field<this_type, VERTEX, TR> const &r,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto Y = topology_type::delta_index(
				topology_type::roate(topology_type::dual(s)));
		auto Z = topology_type::delta_index(
				topology_type::inverse_roate(topology_type::dual(s)));

		return get_value(l, s)
				* (get_value(r, (s - Y) - Z) + get_value(r, (s - Y) + Z)
						+ get_value(r, (s + Y) - Z) + get_value(r, (s + Y) + Z))
				* 0.25;
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, FACE, TL> const &r,
			_Field<this_type, EDGE, TR> const &l,
			index_type s) const ->decltype(get_value(l,s)*get_value(r,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

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

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, WEDGE>,
			_Field<this_type, VOLUME, TL> const &l,
			_Field<this_type, VERTEX, TR> const &r,
			index_type s) const ->decltype(get_value(r,s)*get_value(l,s))
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		get_value(l, s) * (

		get_value(r, ((s - X) - Y) - Z) + //
				get_value(r, ((s - X) - Y) + Z) + //
				get_value(r, ((s - X) + Y) - Z) + //
				get_value(r, ((s - X) + Y) + Z) + //
				get_value(r, ((s + X) - Y) - Z) + //
				get_value(r, ((s + X) - Y) + Z) + //
				get_value(r, ((s + X) + Y) - Z) + //
				get_value(r, ((s + X) + Y) + Z)   //

		) * 0.125;
	}

//***************************************************************************************************

	template<unsigned int IL, typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, HODGESTAR>,
			_Field<this_type, IL, TL> const & f,
			index_type s) const-> typename std::remove_reference<decltype(get_value(f,s))>::type
	{
//		auto X = topology_type::DI(0,s);
//		auto Y = topology_type::DI(1,s);
//		auto Z =topology_type::DI(2,s);
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

	template<typename TL, typename TR> void Opcalculus(
			std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
			nTuple<NDIMS, TR> const & v, _Field<this_type, VERTEX, TL> const & f,
			index_type s) const = delete;

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
			nTuple<NDIMS, TR> const & v, _Field<this_type, EDGE, TL> const & f,
			index_type s) const->decltype(get_value(f,s)*v[0])
	{
		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return

		(get_value(f, s + X) - get_value(f, s - X)) * 0.5 * v[0]  //
		+ (get_value(f, s + Y) - get_value(f, s - Y)) * 0.5 * v[1]   //
		+ (get_value(f, s + Z) - get_value(f, s - Z)) * 0.5 * v[2];
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
			nTuple<NDIMS, TR> const & v, _Field<this_type, FACE, TL> const & f,
			index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = topology_type::component_number(s);

		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);
		return

		(get_value(f, s + Y) + get_value(f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(get_value(f, s + Z) + get_value(f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, INTERIOR_PRODUCT>,
			nTuple<NDIMS, TR> const & v, _Field<this_type, VOLUME, TL> const & f,
			index_type s) const->decltype(get_value(f,s)*v[0])
	{
		unsigned int n = topology_type::component_number(
				topology_type::dual(s));
		unsigned int D = topology_type::delta_index(topology_type::dual(s));

		return (get_value(f, s + D) - get_value(f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation
// For curlpdx

	template<unsigned int N, typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, EXTRIORDERIVATIVE>,
			_Field<this_type, EDGE, TL> const & f,
			std::integral_constant<unsigned int, N>,
			index_type s) const-> decltype(get_value(f,s)-get_value(f,s))
	{

		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		Y = (topology_type::component_number(Y) == N) ? Y : 0UL;
		Z = (topology_type::component_number(Z) == N) ? Z : 0UL;

		return (get_value(f, s + Y) - get_value(f, s - Y))
				- (get_value(f, s + Z) - get_value(f, s - Z));
	}

	template<unsigned int N, typename TL> inline auto Opcalculus(
			std::integral_constant<unsigned int, CODIFFERENTIAL>,
			_Field<this_type, FACE, TL> const & f,
			std::integral_constant<unsigned int, N>,
			index_type s) const-> decltype((get_value(f,s)-get_value(f,s))*std::declval<scalar_type>())
	{

		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		Y = (topology_type::component_number(Y) == N) ? Y : 0UL;
		Z = (topology_type::component_number(Z) == N) ? Z : 0UL;

		return (

		get_value(f, s + Y) * (geo.dual_volume(s + Y))      //
		- get_value(f, s - Y) * (geo.dual_volume(s - Y))    //
		- get_value(f, s + Z) * (geo.dual_volume(s + Z))    //
		+ get_value(f, s - Z) * (geo.dual_volume(s - Z))    //

		) * geo.inv_dual_volume(s);
	}
	template<unsigned int IL, typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, IL> const &,
			_Field<this_type, IL, TR> const & f, index_type s) const
			DECL_RET_TYPE(get_value(f,s))

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, VERTEX> const &,
			_Field<this_type, EDGE, TR> const & f,
			index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return nTuple<3,
				typename std::remove_reference<decltype(get_value(f,s))>::type>(
		{

		(get_value(f, s - X) + get_value(f, s + X)) * 0.5, //
		(get_value(f, s - Y) + get_value(f, s + Y)) * 0.5, //
		(get_value(f, s - Z) + get_value(f, s + Z)) * 0.5

		});
	}

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, EDGE> const &,
			_Field<this_type, VERTEX, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(s);
		auto D = topology_type::delta_index(s);

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, VERTEX> const &,
			_Field<this_type, FACE, TR> const & f,
			index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s))>::type>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

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

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, FACE> const &,
			_Field<this_type, VERTEX, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(topology_type::dual(s));
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

		(

		get_value(f, (s - Y) - Z)[n] +

		get_value(f, (s - Y) + Z)[n] +

		get_value(f, (s + Y) - Z)[n] +

		get_value(f, (s + Y) + Z)[n]

		) * 0.25

		);
	}

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, VOLUME>,
			_Field<this_type, FACE, TR> const & f,
			index_type s) const->nTuple<3,decltype(get_value(f,s) )>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

		return nTuple<3,
				typename std::remove_reference<decltype(get_value(f,s))>::type>(
		{

		(get_value(f, s - X) + get_value(f, s + X)) * 0.5, //
		(get_value(f, s - Y) + get_value(f, s + Y)) * 0.5, //
		(get_value(f, s - Z) + get_value(f, s + Z)) * 0.5

		});
	}

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, FACE>,
			_Field<this_type, VOLUME, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(topology_type::dual(s));
		auto D = topology_type::delta_index(topology_type::dual(s));

		return ((get_value(f, s - D)[n] + get_value(f, s + D)[n]) * 0.5);
	}

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, VOLUME>,
			_Field<this_type, EDGE, TR> const & f,
			index_type s) const->nTuple<3,typename std::remove_reference<decltype(get_value(f,s) )>::type>
	{

		auto X = topology_type::DI(0, s);
		auto Y = topology_type::DI(1, s);
		auto Z = topology_type::DI(2, s);

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

	template<typename TR> inline auto Opcalculus(
			std::integral_constant<unsigned int, MAPTO>,
			std::integral_constant<unsigned int, EDGE>,
			_Field<this_type, VOLUME, TR> const & f,
			index_type s) const->typename std::remove_reference<decltype(get_value(f,s)[0])>::type
	{

		auto n = topology_type::component_number(topology_type::dual(s));
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);
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

#endif /* MESH_RECTANGLE_H_ */
