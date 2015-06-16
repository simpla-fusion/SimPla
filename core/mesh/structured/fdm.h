/**
 * @file  fdm.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef FDM_H_
#define FDM_H_

#include <complex>
#include <cstddef>
#include <type_traits>

#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../calculus.h"
#include "../mesh_ids.h"

namespace simpla
{

template<typename ... > class _Field;

/** @ingroup diff_scheme
 *  @brief   FdMesh
 */
namespace tags
{

struct finite_difference;

}  // namespace tags

namespace policy
{
template<typename TM, typename TAGS> struct calculate;

template<typename MeshType>
struct calculate<MeshType, tags::finite_difference>
{
public:

	typedef MeshType mesh_type;
	typedef calculate<mesh_type, tags::finite_difference> this_type;
	typedef Real scalar_type;

	///***************************************************************************************************
	/// @name general_algebra General algebra
	/// @{
	///***************************************************************************************************

	template<typename ...Others>
	static Real eval(mesh_type const & geo, Real v, Others &&... s)
	{
		return v;
	}

	template<typename ...Others>
	static int eval(mesh_type const & geo, int v, Others &&... s)
	{
		return v;
	}

	template<typename ...Others>
	static std::complex<Real> eval(mesh_type const & geo, std::complex<Real> v,
			Others &&... s)
	{
		return v;
	}

	template<typename T, size_t ...N, typename ...Others>
	static nTuple<T, N...> const& eval(mesh_type const & geo,
			nTuple<T, N...> const& v, Others &&... s)
	{
		return v;
	}

	template<typename ...T, typename ...Others>
	static inline traits::primary_type_t<nTuple<Expression<T...> > > eval(
			mesh_type const & geo, nTuple<Expression<T...>> const & v,
			Others &&... s)
	{
		traits::primary_type_t<nTuple<Expression<T...> > > res;
		res = v;
		return std::move(res);
	}

	template<typename TM, typename TV, typename ... Others, typename ... Args>
	static inline TV eval(mesh_type const & geo,
			_Field<TM, TV, Others...> const &f, Args && ... s)
	{
		return try_index(f, std::forward<Args>(s)...);
	}

	template<typename TOP, typename TL, typename TR, typename ...Others>
	static inline traits::primary_type_t<
			traits::value_type_t<_Field<Expression<TOP, TL, TR>> >>eval(
			mesh_type const & geo, _Field<Expression<TOP, TL, TR>> const &f,
			Others &&... s)
	{
		traits::primary_type_t<
		traits::value_type_t<_Field<Expression<TOP, TL, TR>> >> res;

		res= (f.op_(eval(geo, f.lhs, std::forward<Others>(s)...),
						eval(geo, f.rhs, std::forward<Others>(s)...)));

		return std::move(res);
	}

	template<typename TOP, typename TL,
	typename ...Others>
	static inline traits::primary_type_t<traits::value_type_t<
	_Field<Expression<TOP, TL, std::nullptr_t>> > > eval(
			mesh_type const & geo,
			_Field<Expression<TOP, TL, std::nullptr_t>> const &f,
			Others &&... s)
	{
		traits::primary_type_t<traits::value_type_t<
		_Field<Expression<TOP, TL, std::nullptr_t>> > > res;
		res = f.op_(eval(geo, f.lhs, std::forward<Others>(s)...));

		return std::move(res);
	}

	//***************************************************************************************************
	// Exterior algebra
	//***************************************************************************************************

	template<typename T>
	static inline traits::value_type_t<
	_Field<tags::ExteriorDerivative<VERTEX, T>>> eval(
			mesh_type const & geo,
			_Field<tags::ExteriorDerivative<VERTEX, T> > const & f,
			typename mesh_type::id_type s)
	{
		typename mesh_type::id_type D = mesh_type::delta_index(s);
		return (eval(geo, f.lhs, s + D) * geo.volume(s + D)
				- eval(geo, f.lhs, s - D) * geo.volume(s - D))
		* geo.inv_volume(s);
	}

	template<typename T>
	static inline traits::value_type_t<
	_Field<tags::ExteriorDerivative<EDGE, T>>> eval(
			mesh_type const & geo,
			_Field<tags::ExteriorDerivative<EDGE, T> > const & expr,
			typename mesh_type::id_type s)
	{

		typename mesh_type::id_type X = mesh_type::delta_index(
				mesh_type::dual(s));
		typename mesh_type::id_type Y = mesh_type::rotate(X);
		typename mesh_type::id_type Z = mesh_type::inverse_rotate(X);

		return ((eval(geo, expr.lhs, s + Y) * geo.volume(s + Y) //
						- eval(geo, expr.lhs, s - Y) * geo.volume(s - Y))
				- (eval(geo, expr.lhs, s + Z) * geo.volume(s + Z)//
						- eval(geo, expr.lhs, s - Z) * geo.volume(s - Z)//
				)

		) * geo.inv_volume(s);

	}

	template<typename T>
	static constexpr inline traits::value_type_t<
	_Field<tags::ExteriorDerivative<FACE, T> > > eval(
			mesh_type const & geo,
			_Field<tags::ExteriorDerivative<FACE, T> > const & expr,
			typename mesh_type::id_type s)
	{
		return (eval(geo, expr.lhs, s + mesh_type::_DI)
				* geo.volume(s + mesh_type::_DI)
				- eval(geo, expr.lhs, s - mesh_type::_DI)
				* geo.volume(s - mesh_type::_DI)
				+ eval(geo, expr.lhs, s + mesh_type::_DJ)
				* geo.volume(s + mesh_type::_DJ)
				- eval(geo, expr.lhs, s - mesh_type::_DJ)
				* geo.volume(s - mesh_type::_DJ)
				+ eval(geo, expr.lhs, s + mesh_type::_DK)
				* geo.volume(s + mesh_type::_DK)
				- eval(geo, expr.lhs, s - mesh_type::_DK)
				* geo.volume(s - mesh_type::_DK)

		) * geo.inv_volume(s)

		;
	}
//
////	template<typename geometry_type,typename TM, size_t IL, typename TL> void eval(
////			tags::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
////					typename geometry_type::id_type   s)  = delete;
////
////	template<typename geometry_type,typename TM, size_t IL, typename TL> void eval(
////			tags::CodifferentialDerivative,
////			_Field<TL...> const & f, 		typename geometry_type::id_type   s)  = delete;

	template<typename T>
	static constexpr inline traits::value_type_t<
	_Field<tags::CodifferentialDerivative<EDGE, T> > > eval(
			mesh_type const & geo,
			_Field<tags::CodifferentialDerivative<EDGE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		return -(eval(geo, expr.lhs, s + mesh_type::_DI)
				* geo.dual_volume(s + mesh_type::_DI)
				- eval(geo, expr.lhs, s - mesh_type::_DI)
				* geo.dual_volume(s - mesh_type::_DI)
				+ eval(geo, expr.lhs, s + mesh_type::_DJ)
				* geo.dual_volume(s + mesh_type::_DJ)
				- eval(geo, expr.lhs, s - mesh_type::_DJ)
				* geo.dual_volume(s - mesh_type::_DJ)
				+ eval(geo, expr.lhs, s + mesh_type::_DK)
				* geo.dual_volume(s + mesh_type::_DK)
				- eval(geo, expr.lhs, s - mesh_type::_DK)
				* geo.dual_volume(s - mesh_type::_DK)

		) * geo.inv_dual_volume(s)

		;

	}

	template<typename T>
	static inline traits::value_type_t<
	_Field<tags::CodifferentialDerivative<FACE, T> > > eval(
			mesh_type const & geo,
			_Field<tags::CodifferentialDerivative<FACE, T>> const & expr,
			typename mesh_type::id_type s)
	{

		typename mesh_type::id_type X = mesh_type::delta_index(s);
		typename mesh_type::id_type Y = mesh_type::rotate(X);
		typename mesh_type::id_type Z = mesh_type::inverse_rotate(X);

		return

		-((eval(geo, expr.lhs, s + Y) * (geo.dual_volume(s + Y))
						- eval(geo, expr.lhs, s - Y) * (geo.dual_volume(s - Y)))

				- (eval(geo, expr.lhs, s + Z) * (geo.dual_volume(s + Z))
						- eval(geo, expr.lhs, s - Z)
						* (geo.dual_volume(s - Z)))

		) * geo.inv_dual_volume(s)

		;
	}

	template<typename T>
	static inline traits::value_type_t<
	_Field<tags::CodifferentialDerivative<VOLUME, T> > > eval(
			mesh_type const & geo,
			_Field<tags::CodifferentialDerivative<VOLUME, T>> const & expr,
			typename mesh_type::id_type s)
	{
		typename mesh_type::id_type D = mesh_type::delta_index(
				mesh_type::dual(s));
		return

		-(

				eval(geo, expr.lhs, s + D) * (geo.dual_volume(s + D)) //
				- eval(geo, expr.lhs, s - D) * (geo.dual_volume(s - D))

		) * geo.inv_dual_volume(s)

		;
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<VERTEX, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<VERTEX, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		return (eval(expr.lhs, s) * eval(expr.rhs, s));
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<VERTEX, EDGE, TL, TR> > > eval(
			mesh_type& geo,
			_Field<tags::Wedge<VERTEX, EDGE, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto X = mesh_type::delta_index(s);

		return (eval(expr.lhs, s - X) + eval(expr.lhs, s + X)) * 0.5
		* eval(expr.rhs, s);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<VERTEX, FACE, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<VERTEX, FACE, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto X = mesh_type::delta_index(mesh_type::dual(s));
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);

		return (

				eval(geo, expr.lhs, (s - Y) - Z)
				+ eval(geo, expr.lhs, (s - Y) + Z)
				+ eval(geo, expr.lhs, (s + Y) - Z)
				+ eval(geo, expr.lhs, (s + Y) + Z)

		) * 0.25 * eval(geo, expr.rhs, s);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<VERTEX, VOLUME, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<VERTEX, VOLUME, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{

		auto const & l = expr.lhs;
		auto const & r = expr.rhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return (

				eval(geo, l, ((s - X) - Y) - Z) +

				eval(geo, l, ((s - X) - Y) + Z) +

				eval(geo, l, ((s - X) + Y) - Z) +

				eval(geo, l, ((s - X) + Y) + Z) +

				eval(geo, l, ((s + X) - Y) - Z) +

				eval(geo, l, ((s + X) - Y) + Z) +

				eval(geo, l, ((s + X) + Y) - Z) +

				eval(geo, l, ((s + X) + Y) + Z)

		) * 0.125 * eval(geo, r, s);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<EDGE, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<EDGE, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{

		auto const & l = expr.lhs;
		auto const & r = expr.rhs;

		auto X = mesh_type::delta_index(s);
		return eval(geo, l, s)
		* (eval(geo, r, s - X) + eval(geo, r, s + X)) * 0.5;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<tags::Wedge<EDGE, EDGE, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<EDGE, EDGE, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & l = expr.lhs;
		auto const & r = expr.rhs;

		auto Y = mesh_type::delta_index(
				mesh_type::rotate(mesh_type::dual(s)));
		auto Z = mesh_type::delta_index(
				mesh_type::inverse_rotate(mesh_type::dual(s)));

		return ((eval(geo, l, s - Y) + eval(geo, l, s + Y))
				* (eval(geo, l, s - Z) + eval(geo, l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<tags::Wedge<EDGE, FACE, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<EDGE, FACE, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & l = expr.lhs;
		auto const & r = expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

				(eval(geo, l, (s - Y) - Z) + eval(geo, l, (s - Y) + Z)
						+ eval(geo, l, (s + Y) - Z)
						+ eval(geo, l, (s + Y) + Z))
				* (eval(geo, r, s - X) + eval(geo, r, s + X))
				+

				(eval(geo, l, (s - Z) - X) + eval(geo, l, (s - Z) + X)
						+ eval(geo, l, (s + Z) - X)
						+ eval(geo, l, (s + Z) + X))
				* (eval(geo, r, s - Y) + eval(geo, r, s + Y))
				+

				(eval(geo, l, (s - X) - Y) + eval(geo, l, (s - X) + Y)
						+ eval(geo, l, (s + X) - Y)
						+ eval(geo, l, (s + X) + Y))
				* (eval(geo, r, s - Z) + eval(geo, r, s + Z))

		) * 0.125;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<FACE, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<FACE, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & l = expr.lhs;
		auto const & r = expr.rhs;
		auto Y = mesh_type::delta_index(
				mesh_type::rotate(mesh_type::dual(s)));
		auto Z = mesh_type::delta_index(
				mesh_type::inverse_rotate(mesh_type::dual(s)));

		return eval(geo, l, s)
		* (eval(geo, r, (s - Y) - Z)
				+ eval(geo, r, (s - Y) + Z)
				+ eval(geo, r, (s + Y) - Z)
				+ eval(geo, r, (s + Y) + Z)) * 0.25;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<tags::Wedge<FACE, EDGE, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<FACE, EDGE, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & l = expr.lhs;
		auto const & r = expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

				(eval(geo, r, (s - Y) - Z) + eval(geo, r, (s - Y) + Z)
						+ eval(geo, r, (s + Y) - Z)
						+ eval(geo, r, (s + Y) + Z))
				* (eval(geo, l, s - X) + eval(geo, l, s + X))

				+ (eval(geo, r, (s - Z) - X)
						+ eval(geo, r, (s - Z) + X)
						+ eval(geo, r, (s + Z) - X)
						+ eval(geo, r, (s + Z) + X))
				* (eval(geo, l, s - Y) + eval(geo, l, s + Y))

				+ (eval(geo, r, (s - X) - Y)
						+ eval(geo, r, (s - X) + Y)
						+ eval(geo, r, (s + X) - Y)
						+ eval(geo, r, (s + X) + Y))
				* (eval(geo, l, s - Z) + eval(geo, l, s + Z))

		) * 0.125;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::Wedge<VOLUME, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::Wedge<VOLUME, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & l = expr.lhs;
		auto const & r = expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		eval(geo, l, s) * (

				eval(geo, r, ((s - X) - Y) - Z) + //
				eval(geo, r, ((s - X) - Y) + Z) +//
				eval(geo, r, ((s - X) + Y) - Z) +//
				eval(geo, r, ((s - X) + Y) + Z) +//
				eval(geo, r, ((s + X) - Y) - Z) +//
				eval(geo, r, ((s + X) - Y) + Z) +//
				eval(geo, r, ((s + X) + Y) - Z) +//
				eval(geo, r, ((s + X) + Y) + Z)//

		) * 0.125;
	}
//
////***************************************************************************************************

	template<typename T>
	static inline traits::value_type_t<_Field<tags::HodgeStar<VOLUME, T> > > eval(
			mesh_type const & geo,
			_Field<tags::HodgeStar<VOLUME, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
//		auto X = geo.DI(0,s);
//		auto Y = geo.DI(1,s);
//		auto Z =geo.DI(2,s);
//
//		return
//
//		(
//
//		eval(geo,f,((s + X) - Y) - Z)*geo.inv_volume(((s + X) - Y) - Z) +
//
//		eval(geo,f,((s + X) - Y) + Z)*geo.inv_volume(((s + X) - Y) + Z) +
//
//		eval(geo,f,((s + X) + Y) - Z)*geo.inv_volume(((s + X) + Y) - Z) +
//
//		eval(geo,f,((s + X) + Y) + Z)*geo.inv_volume(((s + X) + Y) + Z) +
//
//		eval(geo,f,((s - X) - Y) - Z)*geo.inv_volume(((s - X) - Y) - Z) +
//
//		eval(geo,f,((s - X) - Y) + Z)*geo.inv_volume(((s - X) - Y) + Z) +
//
//		eval(geo,f,((s - X) + Y) - Z)*geo.inv_volume(((s - X) + Y) - Z) +
//
//		eval(geo,f,((s - X) + Y) + Z)*geo.inv_volume(((s - X) + Y) + Z)
//
//		) * 0.125 * geo.volume(s);

		return eval(geo, f, s) /** geo.tags::HodgeStarVolumeScale(s)*/;
	}

//	template<typename geometry_type,typename TM, typename TL, typename TR> void eval(
//			tags::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, VERTEX>, TL> const & f,
//					typename geometry_type::id_type   s)  = delete;

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::InteriorProduct<EDGE, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::InteriorProduct<EDGE, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
		auto const & v = expr.rhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(eval(geo, f, s + X) - eval(geo, f, s - X)) * 0.5 * v[0] //
		+ (eval(geo, f, s + Y) - eval(geo, f, s - Y)) * 0.5 * v[1]//
		+ (eval(geo, f, s + Z) - eval(geo, f, s - Z)) * 0.5 * v[2];
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::InteriorProduct<FACE, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::InteriorProduct<FACE, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
		auto const & v = expr.rhs;

		size_t n = mesh_type::component_number(s);

		auto X = mesh_type::delta_index(s);
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);
		return

		(eval(geo, f, s + Y) + eval(geo, f, s - Y)) * 0.5
		* v[(n + 2) % 3]
		-

		(eval(geo, f, s + Z) + eval(geo, f, s - Z)) * 0.5
		* v[(n + 1) % 3];
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<
	_Field<tags::InteriorProduct<VOLUME, VERTEX, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<tags::InteriorProduct<VOLUME, VERTEX, TL, TR>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
		auto const & v = expr.rhs;
		size_t n = mesh_type::component_number(mesh_type::dual(s));
		size_t D = mesh_type::delta_index(mesh_type::dual(s));

		return (eval(geo, f, s + D) - eval(geo, f, s - D)) * 0.5
		* v[n];
	}

//**************************************************************************************************
// Non-standard operation

	template<size_t IL, typename T> static inline traits::value_type_t<
	T> eval(mesh_type const & geo,
			_Field<tags::MapTo<IL, IL, T>> const & f,
			typename mesh_type::id_type s)
	{
		return eval(geo, f, s);
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<EDGE, VERTEX, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<EDGE, VERTEX, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
		3>(
				{

					(eval(geo, f, s - X) + eval(geo, f, s + X)) * 0.5, //
					(eval(geo, f, s - Y) + eval(geo, f, s + Y)) * 0.5,//
					(eval(geo, f, s - Z) + eval(geo, f, s + Z)) * 0.5

				});
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<VERTEX, EDGE, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<VERTEX, EDGE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
		auto n = mesh_type::component_number(s);
		auto D = mesh_type::delta_index(s);

		return ((eval(geo, f, s - D)[n] + eval(geo, f, s + D)[n])
				* 0.5);
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<FACE, VERTEX, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<FACE, VERTEX, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
		3>(
				{	(

							eval(geo, f, (s - Y) - Z) +

							eval(geo, f, (s - Y) + Z) +

							eval(geo, f, (s + Y) - Z) +

							eval(geo, f, (s + Y) + Z)

					) * 0.25,

					(

							eval(geo, f, (s - Z) - X) +

							eval(geo, f, (s - Z) + X) +

							eval(geo, f, (s + Z) - X) +

							eval(geo, f, (s + Z) + X)

					) * 0.25,

					(

							eval(geo, f, (s - X) - Y) +

							eval(geo, f, (s - X) + Y) +

							eval(geo, f, (s + X) - Y) +

							eval(geo, f, (s + X) + Y)

					) * 0.25

				});
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<VERTEX, FACE, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<VERTEX, FACE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto n = mesh_type::component_number(mesh_type::dual(s));
		auto X = mesh_type::delta_index(mesh_type::dual(s));
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);

		return (

				(

						eval(geo, f, (s - Y) - Z)[n] +

						eval(geo, f, (s - Y) + Z)[n] +

						eval(geo, f, (s + Y) - Z)[n] +

						eval(geo, f, (s + Y) + Z)[n]

				) * 0.25

		);
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<FACE, VOLUME, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<FACE, VOLUME, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
		3>(
				{

					(eval(geo, f, s - X) + eval(geo, f, s + X)) * 0.5, //
					(eval(geo, f, s - Y) + eval(geo, f, s + Y)) * 0.5,//
					(eval(geo, f, s - Z) + eval(geo, f, s + Z)) * 0.5

				});
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<VOLUME, FACE, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<VOLUME, FACE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto n = mesh_type::component_number(mesh_type::dual(s));
		auto D = mesh_type::delta_index(mesh_type::dual(s));

		return ((eval(geo, f, s - D)[n] + eval(geo, f, s + D)[n])
				* 0.5);
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<EDGE, VOLUME, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<EDGE, VOLUME, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
		3>(
				{	(

							eval(geo, f, (s - Y) - Z) +

							eval(geo, f, (s - Y) + Z) +

							eval(geo, f, (s + Y) - Z) +

							eval(geo, f, (s + Y) + Z)

					) * 0.25,

					(

							eval(geo, f, (s - Z) - X) +

							eval(geo, f, (s - Z) + X) +

							eval(geo, f, (s + Z) - X) +

							eval(geo, f, (s + Z) + X)

					) * 0.25,

					(

							eval(geo, f, (s - X) - Y) +

							eval(geo, f, (s - X) + Y) +

							eval(geo, f, (s + X) - Y) +

							eval(geo, f, (s + X) + Y)

					) * 0.25,

				});
	}

	template<typename T>
	traits::value_type_t<_Field<tags::MapTo<VOLUME, EDGE, T>> > map_to(
			mesh_type const & geo,
			_Field<tags::MapTo<VOLUME, EDGE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;
		auto n = mesh_type::component_number(mesh_type::dual(s));
		auto X = mesh_type::delta_index(mesh_type::dual(s));
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);
		return (

				(

						eval(geo, f, (s - Y) - Z)[n] +

						eval(geo, f, (s - Y) + Z)[n] +

						eval(geo, f, (s + Y) - Z)[n] +

						eval(geo, f, (s + Y) + Z)[n]

				) * 0.25

		);
	}

	// For curl_pdx

	template<size_t N, typename T> static inline traits::value_type_t<
	_Field<tags::PartialExteriorDerivative<N, EDGE, T> > > eval(
			mesh_type const & geo,
			_Field<tags::PartialExteriorDerivative<N, EDGE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto X = mesh_type::delta_index(mesh_type::dual(s));
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);

		Y = (mesh_type::component_number(Y) == N) ? Y : 0UL;
		Z = (mesh_type::component_number(Z) == N) ? Z : 0UL;

		return (eval(geo, f, s + Y) - eval(geo, f, s - Y))
		- (eval(geo, f, s + Z) - eval(geo, f, s - Z));
	}

	template<size_t N, typename T> static inline traits::value_type_t<
	_Field<tags::PartialCodifferentialDerivative<N, FACE, T>> > eval(
			mesh_type const & geo,
			_Field<tags::PartialCodifferentialDerivative<N, FACE, T>> const & expr,
			typename mesh_type::id_type s)
	{
		auto const & f = expr.lhs;

		auto X = mesh_type::delta_index(s);
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);

		Y = (mesh_type::component_number(Y) == N) ? Y : 0UL;
		Z = (mesh_type::component_number(Z) == N) ? Z : 0UL;

		return (

				eval(geo, f, s + Y) * (geo.dual_volume(s + Y))      //
				- eval(geo, f, s - Y) * (geo.dual_volume(s - Y))//
				- eval(geo, f, s + Z) * (geo.dual_volume(s + Z))//
				+ eval(geo, f, s - Z) * (geo.dual_volume(s - Z))//

		) * geo.inv_dual_volume(s);
	}

};
}
      // namespace policy
}
// namespace simpla

#endif /* FDM_H_ */
