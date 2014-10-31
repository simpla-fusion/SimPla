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
	typedef G geometry_type;
	typedef typename geometry_type::compact_index_type compact_index_type;
	typedef Real scalar_type;
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

//	template<typename ...Args>
//	auto calculate(Args && ... args) const
//	DECL_RET_TYPE( (get_value( std::forward<Args>(args)...)) )

	template<typename TC, typename TD>
	inline auto calculate(_Field<TC, TD> const& f, compact_index_type s) const
	DECL_RET_TYPE((f[s] ) )

	template<typename TOP, typename TL>
	auto calculate(_Field<Expression<TOP, TL> > const & f,
			compact_index_type s) const
			DECL_RET_TYPE((this->calculate(f.op_,f.lhs,s)))

	template<typename TOP, typename TL, typename TR>
	auto calculate(_Field<Expression<TOP, TL, TR> > const & f,
			compact_index_type s) const
			DECL_RET_TYPE((this->calculate(f.op_,f.lhs,f.rhs,s)))

	template<typename T>
	auto calculate(T const & v, compact_index_type s) const
	DECL_RET_TYPE((get_value(v,s)))

	template<typename TOP, typename TL>
	inline auto calculate(TOP op, TL const& f, compact_index_type s) const
	DECL_RET_TYPE( op(this->calculate(f,s) ) )

	template<typename TOP, typename TL, typename TR>
	inline auto calculate(TOP op, TL const& l, TR const &r,
			compact_index_type s) const
			DECL_RET_TYPE( op(this->calculate( (l),s),this->calculate(r,s) ) )

	template<typename ... TL>
	inline auto calculate(_impl::ExteriorDerivative, _Field<TL...> const & f,
			compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==VERTEX,Real
			/*decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())*/>::type
	{
		auto D = geo->delta_index(s);

		return (this->calculate(f, s + D) * geo->volume(s + D)
				- this->calculate(f, s - D) * geo->volume(s - D))
				* geo->inv_volume(s);
	}

	template<typename ...TL>
	inline auto calculate(_impl::ExteriorDerivative, _Field<TL...> const & f,
			compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==EDGE,Real
			/*decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())*/
			>::type
	{
		auto X = geo->delta_index(geo->dual(s));
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return (

		(

		this->calculate(f, s + Y) * geo->volume(s + Y) //
		- this->calculate(f, s - Y) * geo->volume(s - Y) //

		) - (

		this->calculate(f, s + Z) * geo->volume(s + Z) //
		- this->calculate(f, s - Z) * geo->volume(s - Z) //

		)

		) * geo->inv_volume(s);

	}

	template<typename ... TL>
	inline auto calculate(_impl::ExteriorDerivative, _Field<TL...> const & f,
			compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==FACE,Real
//			decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())
			>::type
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return (

		this->calculate(f, s + X) * geo->volume(s + X)

		- this->calculate(f, s - X) * geo->volume(s - X) //
		+ this->calculate(f, s + Y) * geo->volume(s + Y) //
		- this->calculate(f, s - Y) * geo->volume(s - Y) //
		+ this->calculate(f, s + Z) * geo->volume(s + Z) //
		- this->calculate(f, s - Z) * geo->volume(s - Z) //

		) * geo->inv_volume(s)

		;
	}

//	template<typename TM, size_t IL, typename TL> void calculate(
//			_impl::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
//			compact_index_type s) const = delete;
//
//	template<typename TM, size_t IL, typename TL> void calculate(
//			_impl::CodifferentialDerivative,
//			_Field<TL...> const & f, compact_index_type s) const = delete;

	template<typename ... TL> inline auto calculate(
			_impl::CodifferentialDerivative, _Field<TL...> const & f,
			compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==EDGE,Real
//			decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())
			>::type
	{
		auto X = geo->DI(0, s);
		auto Y = geo->DI(1, s);
		auto Z = geo->DI(2, s);

		return

		-(

		this->calculate(f, s + X) * geo->dual_volume(s + X)

		- this->calculate(f, s - X) * geo->dual_volume(s - X)

		+ this->calculate(f, s + Y) * geo->dual_volume(s + Y)

		- this->calculate(f, s - Y) * geo->dual_volume(s - Y)

		+ this->calculate(f, s + Z) * geo->dual_volume(s + Z)

		- this->calculate(f, s - Z) * geo->dual_volume(s - Z)

		) * geo->inv_dual_volume(s)

		;

	}

	template<typename ...TL>
	inline auto calculate(_impl::CodifferentialDerivative,
			_Field<TL...> const & f,
			compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==FACE,Real
//			decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())
			>::type
	{
		auto X = geo->delta_index(s);
		auto Y = geo->roate(X);
		auto Z = geo->inverse_roate(X);

		return

		-(

		(this->calculate(f, s + Y) * (geo->dual_volume(s + Y))
				- this->calculate(f, s - Y) * (geo->dual_volume(s - Y)))

				- (this->calculate(f, s + Z) * (geo->dual_volume(s + Z))
						- this->calculate(f, s - Z) * (geo->dual_volume(s - Z)))

		) * geo->inv_dual_volume(s)

		;
	}

	template<typename ...TL> inline auto calculate(
			_impl::CodifferentialDerivative, _Field<TL...> const & f,
			compact_index_type s) const ->
			typename std::enable_if<field_traits<_Field<TL...>>::iform==VOLUME,Real
//			decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())
			>::type
	{
		auto D = geo->delta_index(geo->dual(s));
		return

		-(

		this->calculate(f, s + D) * (geo->dual_volume(s + D)) //
		- this->calculate(f, s - D) * (geo->dual_volume(s - D))

		) * geo->inv_dual_volume(s)

		;
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>
//	template<typename ... TL, typename ... TR> inline auto calculate(
//			_impl::Wedge, _Field<TL...> const &l, _Field<TR...> const &r,
//			compact_index_type s) const->
//			typename std::enable_if<
//			field_traits<_Field<TL...>>::iform==VERTEX &&
//			field_traits<_Field<TR...>>::iform==VERTEX,
//			decltype(this->calculate(l,s)*this->calculate(r,s))>::type
//	{
//		return this->calculate(l, s) * this->calculate(r, s);
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
//			_Field<Domain<TM, EDGE>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto X = geo->delta_index(s);
//
//		return (this->calculate(l, s - X) + this->calculate(l, s + X)) * 0.5
//				* this->calculate(r, s);
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
//			_Field<Domain<TM, FACE>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto X = geo->delta_index(geo->dual(s));
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//
//		return (
//
//		this->calculate(l, (s - Y) - Z) +
//
//		this->calculate(l, (s - Y) + Z) +
//
//		this->calculate(l, (s + Y) - Z) +
//
//		this->calculate(l, (s + Y) + Z)
//
//		) * 0.25 * this->calculate(r, s);
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, VERTEX>, TL> const &l,
//			_Field<Domain<TM, VOLUME>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return (
//
//		this->calculate(l, ((s - X) - Y) - Z) +
//
//		this->calculate(l, ((s - X) - Y) + Z) +
//
//		this->calculate(l, ((s - X) + Y) - Z) +
//
//		this->calculate(l, ((s - X) + Y) + Z) +
//
//		this->calculate(l, ((s + X) - Y) - Z) +
//
//		this->calculate(l, ((s + X) - Y) + Z) +
//
//		this->calculate(l, ((s + X) + Y) - Z) +
//
//		this->calculate(l, ((s + X) + Y) + Z)
//
//		) * 0.125 * this->calculate(r, s);
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
//			_Field<Domain<TM, VERTEX>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto X = geo->delta_index(s);
//		return this->calculate(l, s) * (this->calculate(r, s - X) + this->calculate(r, s + X))
//				* 0.5;
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
//			_Field<Domain<TM, EDGE>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto Y = geo->delta_index(geo->roate(geo->dual(s)));
//		auto Z = geo->delta_index(geo->inverse_roate(geo->dual(s)));
//
//		return ((this->calculate(l, s - Y) + this->calculate(l, s + Y))
//				* (this->calculate(l, s - Z) + this->calculate(l, s + Z)) * 0.25);
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, EDGE>, TL> const &l,
//			_Field<Domain<TM, FACE>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return
//
//		(
//
//		(this->calculate(l, (s - Y) - Z) + this->calculate(l, (s - Y) + Z)
//				+ this->calculate(l, (s + Y) - Z) + this->calculate(l, (s + Y) + Z))
//				* (this->calculate(r, s - X) + this->calculate(r, s + X))
//				+
//
//				(this->calculate(l, (s - Z) - X) + this->calculate(l, (s - Z) + X)
//						+ this->calculate(l, (s + Z) - X)
//						+ this->calculate(l, (s + Z) + X))
//						* (this->calculate(r, s - Y) + this->calculate(r, s + Y))
//				+
//
//				(this->calculate(l, (s - X) - Y) + this->calculate(l, (s - X) + Y)
//						+ this->calculate(l, (s + X) - Y)
//						+ this->calculate(l, (s + X) + Y))
//						* (this->calculate(r, s - Z) + this->calculate(r, s + Z))
//
//		) * 0.125;
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, FACE>, TL> const &l,
//			_Field<Domain<TM, VERTEX>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto Y = geo->delta_index(geo->roate(geo->dual(s)));
//		auto Z = geo->delta_index(geo->inverse_roate(geo->dual(s)));
//
//		return this->calculate(l, s)
//				* (this->calculate(r, (s - Y) - Z) + this->calculate(r, (s - Y) + Z)
//						+ this->calculate(r, (s + Y) - Z)
//						+ this->calculate(r, (s + Y) + Z)) * 0.25;
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, FACE>, TL> const &r,
//			_Field<Domain<TM, EDGE>, TR> const &l,
//			compact_index_type s) const->decltype(this->calculate(l,s)*this->calculate(r,s))
//	{
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return
//
//		(
//
//		(this->calculate(r, (s - Y) - Z) + this->calculate(r, (s - Y) + Z)
//				+ this->calculate(r, (s + Y) - Z) + this->calculate(r, (s + Y) + Z))
//				* (this->calculate(l, s - X) + this->calculate(l, s + X))
//
//				+ (this->calculate(r, (s - Z) - X) + this->calculate(r, (s - Z) + X)
//						+ this->calculate(r, (s + Z) - X)
//						+ this->calculate(r, (s + Z) + X))
//						* (this->calculate(l, s - Y) + this->calculate(l, s + Y))
//
//				+ (this->calculate(r, (s - X) - Y) + this->calculate(r, (s - X) + Y)
//						+ this->calculate(r, (s + X) - Y)
//						+ this->calculate(r, (s + X) + Y))
//						* (this->calculate(l, s - Z) + this->calculate(l, s + Z))
//
//		) * 0.125;
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::Wedge, _Field<Domain<TM, VOLUME>, TL> const &l,
//			_Field<Domain<TM, VERTEX>, TR> const &r,
//			compact_index_type s) const->decltype(this->calculate(r,s)*this->calculate(l,s))
//	{
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return
//
//		this->calculate(l, s) * (
//
//		this->calculate(r, ((s - X) - Y) - Z) + //
//				this->calculate(r, ((s - X) - Y) + Z) + //
//				this->calculate(r, ((s - X) + Y) - Z) + //
//				this->calculate(r, ((s - X) + Y) + Z) + //
//				this->calculate(r, ((s + X) - Y) - Z) + //
//				this->calculate(r, ((s + X) - Y) + Z) + //
//				this->calculate(r, ((s + X) + Y) - Z) + //
//				this->calculate(r, ((s + X) + Y) + Z) //
//
//		) * 0.125;
//	}
//
////***************************************************************************************************
//
//	template<typename TM, size_t IL, typename TL> inline auto calculate(
//			_impl::HodgeStar, _Field<Domain<TM, IL>, TL> const & f,
//			compact_index_type s) const -> typename std::remove_reference<decltype(this->calculate(f,s))>::type
//	{
////		auto X = geo->DI(0,s);
////		auto Y = geo->DI(1,s);
////		auto Z =geo->DI(2,s);
////
////		return
////
////		(
////
////		this->calculate(f,((s + X) - Y) - Z)*geo->inv_volume(((s + X) - Y) - Z) +
////
////		this->calculate(f,((s + X) - Y) + Z)*geo->inv_volume(((s + X) - Y) + Z) +
////
////		this->calculate(f,((s + X) + Y) - Z)*geo->inv_volume(((s + X) + Y) - Z) +
////
////		this->calculate(f,((s + X) + Y) + Z)*geo->inv_volume(((s + X) + Y) + Z) +
////
////		this->calculate(f,((s - X) - Y) - Z)*geo->inv_volume(((s - X) - Y) - Z) +
////
////		this->calculate(f,((s - X) - Y) + Z)*geo->inv_volume(((s - X) - Y) + Z) +
////
////		this->calculate(f,((s - X) + Y) - Z)*geo->inv_volume(((s - X) + Y) - Z) +
////
////		this->calculate(f,((s - X) + Y) + Z)*geo->inv_volume(((s - X) + Y) + Z)
////
////		) * 0.125 * geo->volume(s);
//
//		return this->calculate(f, s) /** geo->_impl::HodgeStarVolumeScale(s)*/;
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
//			compact_index_type s) const ->decltype(this->calculate(f,s)*v[0])
//	{
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return
//
//		(this->calculate(f, s + X) - this->calculate(f, s - X)) * 0.5 * v[0] //
//		+ (this->calculate(f, s + Y) - this->calculate(f, s - Y)) * 0.5 * v[1] //
//		+ (this->calculate(f, s + Z) - this->calculate(f, s - Z)) * 0.5 * v[2];
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, FACE>, TL> const & f,
//			compact_index_type s) const ->decltype(this->calculate(f,s)*v[0])
//	{
//		size_t n = geo->component_number(s);
//
//		auto X = geo->delta_index(s);
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//		return
//
//		(this->calculate(f, s + Y) + this->calculate(f, s - Y)) * 0.5 * v[(n + 2) % 3] -
//
//		(this->calculate(f, s + Z) + this->calculate(f, s - Z)) * 0.5 * v[(n + 1) % 3];
//	}
//
//	template<typename TM, typename TL, typename TR> inline auto calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, VOLUME>, TL> const & f,
//			compact_index_type s) const ->decltype(this->calculate(f,s)*v[0])
//	{
//		size_t n = geo->component_number(geo->dual(s));
//		size_t D = geo->delta_index(geo->dual(s));
//
//		return (this->calculate(f, s + D) - this->calculate(f, s - D)) * 0.5 * v[n];
//	}
//
////**************************************************************************************************
//// Non-standard operation
//// For curlpdx
//
//	template<typename TM, size_t N, typename TL> inline auto calculate(
//			_impl::ExteriorDerivative, _Field<Domain<TM, EDGE>, TL> const & f,
//			std::integral_constant<size_t, N>,
//			compact_index_type s) const -> decltype(this->calculate(f,s)-this->calculate(f,s))
//	{
//
//		auto X = geo->delta_index(geo->dual(s));
//		auto Y = geo->roate(X);
//		auto Z = geo->inverse_roate(X);
//
//		Y = (geo->component_number(Y) == N) ? Y : 0UL;
//		Z = (geo->component_number(Z) == N) ? Z : 0UL;
//
//		return (this->calculate(f, s + Y) - this->calculate(f, s - Y))
//				- (this->calculate(f, s + Z) - this->calculate(f, s - Z));
//	}
//
//	template<typename ...TL, size_t N> inline auto calculate(
//			_impl::CodifferentialDerivative, _Field<TL...> const & f,
//			std::integral_constant<size_t, N>,
//			compact_index_type s) const ->
//			typename std::enable_if<field_traits<_Field<TL...>>::iform==FACE,
//			decltype((this->calculate(f,s)-this->calculate(f,s))*std::declval<scalar_type>())>::type
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
//		this->calculate(f, s + Y) * (geo->dual_volume(s + Y))      //
//		- this->calculate(f, s - Y) * (geo->dual_volume(s - Y))    //
//		- this->calculate(f, s + Z) * (geo->dual_volume(s + Z))    //
//		+ this->calculate(f, s - Z) * (geo->dual_volume(s - Z))    //
//
//		) * geo->inv_dual_volume(s);
//	}
//	template<typename TM, size_t IL, typename TR> inline auto calculate(
//			_impl::MapTo, std::integral_constant<size_t, IL> const &,
//			_Field<Domain<TM, IL>, TR> const & f, compact_index_type s) const
//			DECL_RET_TYPE(this->calculate(f,s))
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, VERTEX> const &,
//			_Field<Domain<TM, EDGE>, TR> const & f,
//			compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(this->calculate(f,s))>::type,3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(this->calculate(f,s))>::type,
//				3>(
//		{
//
//		(this->calculate(f, s - X) + this->calculate(f, s + X)) * 0.5, //
//		(this->calculate(f, s - Y) + this->calculate(f, s + Y)) * 0.5, //
//		(this->calculate(f, s - Z) + this->calculate(f, s + Z)) * 0.5
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, EDGE> const &,
//			_Field<Domain<TM, VERTEX>, TR> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(this->calculate(f,s)[0])>::type
//	{
//
//		auto n = geo->component_number(s);
//		auto D = geo->delta_index(s);
//
//		return ((this->calculate(f, s - D)[n] + this->calculate(f, s + D)[n]) * 0.5);
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, VERTEX> const &,
//			_Field<Domain<TM, FACE>, TR> const & f,
//			compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(this->calculate(f,s))>::type,3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(this->calculate(f,s))>::type,
//				3>(
//		{ (
//
//		this->calculate(f, (s - Y) - Z) +
//
//		this->calculate(f, (s - Y) + Z) +
//
//		this->calculate(f, (s + Y) - Z) +
//
//		this->calculate(f, (s + Y) + Z)
//
//		) * 0.25,
//
//		(
//
//		this->calculate(f, (s - Z) - X) +
//
//		this->calculate(f, (s - Z) + X) +
//
//		this->calculate(f, (s + Z) - X) +
//
//		this->calculate(f, (s + Z) + X)
//
//		) * 0.25,
//
//		(
//
//		this->calculate(f, (s - X) - Y) +
//
//		this->calculate(f, (s - X) + Y) +
//
//		this->calculate(f, (s + X) - Y) +
//
//		this->calculate(f, (s + X) + Y)
//
//		) * 0.25
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, FACE> const &,
//			_Field<Domain<TM, VERTEX>, TR> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(this->calculate(f,s)[0])>::type
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
//		this->calculate(f, (s - Y) - Z)[n] +
//
//		this->calculate(f, (s - Y) + Z)[n] +
//
//		this->calculate(f, (s + Y) - Z)[n] +
//
//		this->calculate(f, (s + Y) + Z)[n]
//
//		) * 0.25
//
//		);
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, VOLUME>,
//			_Field<Domain<TM, FACE>, TR> const & f,
//			compact_index_type s) const ->nTuple<decltype(this->calculate(f,s) ),3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(this->calculate(f,s))>::type,
//				3>(
//		{
//
//		(this->calculate(f, s - X) + this->calculate(f, s + X)) * 0.5, //
//		(this->calculate(f, s - Y) + this->calculate(f, s + Y)) * 0.5, //
//		(this->calculate(f, s - Z) + this->calculate(f, s + Z)) * 0.5
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, FACE>,
//			_Field<Domain<TM, VOLUME>, TR> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(this->calculate(f,s)[0])>::type
//	{
//
//		auto n = geo->component_number(geo->dual(s));
//		auto D = geo->delta_index(geo->dual(s));
//
//		return ((this->calculate(f, s - D)[n] + this->calculate(f, s + D)[n]) * 0.5);
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, VOLUME>,
//			_Field<Domain<TM, EDGE>, TR> const & f,
//			compact_index_type s) const ->nTuple<typename std::remove_reference<decltype(this->calculate(f,s) )>::type,3>
//	{
//
//		auto X = geo->DI(0, s);
//		auto Y = geo->DI(1, s);
//		auto Z = geo->DI(2, s);
//
//		return nTuple<
//				typename std::remove_reference<decltype(this->calculate(f,s))>::type,
//				3>(
//		{ (
//
//		this->calculate(f, (s - Y) - Z) +
//
//		this->calculate(f, (s - Y) + Z) +
//
//		this->calculate(f, (s + Y) - Z) +
//
//		this->calculate(f, (s + Y) + Z)
//
//		) * 0.25,
//
//		(
//
//		this->calculate(f, (s - Z) - X) +
//
//		this->calculate(f, (s - Z) + X) +
//
//		this->calculate(f, (s + Z) - X) +
//
//		this->calculate(f, (s + Z) + X)
//
//		) * 0.25,
//
//		(
//
//		this->calculate(f, (s - X) - Y) +
//
//		this->calculate(f, (s - X) + Y) +
//
//		this->calculate(f, (s + X) - Y) +
//
//		this->calculate(f, (s + X) + Y)
//
//		) * 0.25,
//
//		});
//	}
//
//	template<typename TM, typename TR> inline auto calculate(_impl::MapTo,
//			std::integral_constant<size_t, EDGE>,
//			_Field<TR, Domain<TM, VOLUME>> const & f,
//			compact_index_type s) const ->typename std::remove_reference<decltype(this->calculate(f,s)[0])>::type
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
//		this->calculate(f, (s - Y) - Z)[n] +
//
//		this->calculate(f, (s - Y) + Z)[n] +
//
//		this->calculate(f, (s + Y) - Z)[n] +
//
//		this->calculate(f, (s + Y) + Z)[n]
//
//		) * 0.25
//
//		);
//	}
public:

};

}
// namespace simpla

#endif /* FDM_H_ */
