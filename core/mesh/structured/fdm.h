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

#include "../../field/calculus.h"
#include "../../field/field_expression.h"
#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../mesh_ids.h"
#include "../mesh_traits.h"

namespace simpla
{

template<typename ... > class _Field;

/** @ingroup diff_scheme
 *  @brief   FdMesh
 */
namespace tags
{

struct finite_difference;

struct HodgeStar;
struct InteriorProduct;
struct Wedge;

struct ExteriorDerivative;
struct CodifferentialDerivative;

struct MapTo;

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

	typedef traits::scalar_type_t<mesh_type> scalar_type;

	typedef traits::id_type_t<mesh_type> id_type;
	///***************************************************************************************************
	/// @name general_algebra General algebra
	/// @{
	///***************************************************************************************************

	static Real eval(mesh_type const & geo, Real v, id_type s)
	{
		return v;
	}

	static int eval(mesh_type const & geo, int v, id_type s)
	{
		return v;
	}

	static std::complex<Real> eval(mesh_type const & geo, std::complex<Real> v,
			id_type s)
	{
		return v;
	}

	template<typename T, size_t ...N>
	static nTuple<T, N...> const& eval(mesh_type const & geo,
			nTuple<T, N...> const& v, id_type s)
	{

		return v;
	}

	template<typename ...T>
	static inline traits::primary_type_t<nTuple<Expression<T...> > > eval(
			mesh_type const & geo, nTuple<Expression<T...>> const & v,
			id_type s)
	{
		traits::primary_type_t<nTuple<Expression<T...> > > res;
		res = v;
		return std::move(res);
	}

	template<typename TM, typename TV, typename ... Others, typename ... Args>
	static inline TV eval(mesh_type const & geo,
			_Field<TM, TV, Others...> const &f, id_type s)
	{
		return try_index(f, s);
	}

	template<typename TOP, typename ... T>
	static traits::primary_type_t<
			traits::value_type_t<_Field<Expression<TOP, T...> > > > eval(
			mesh_type const & geo, _Field<Expression<TOP, T...> > const &expr,
			id_type const &s)
	{
		return eval(geo, expr, s, traits::iform_list_t<T...>());
	}

private:

	template<typename Expr, size_t ... index>
	static traits::primary_type_t<traits::value_type_t<Expr>> _invoke_helper(
			mesh_type const & geo, Expr const & expr, id_type s,
			index_sequence<index...>)
	{
		traits::primary_type_t<traits::value_type_t<Expr>> res = (expr.m_op_(
				eval(geo, std::get<index>(expr.args),s)...));

		return std::move(res);
	}

public:

	template<typename TOP, typename ... T>
	static traits::primary_type_t<
			traits::value_type_t<_Field<Expression<TOP, T...> > > > eval(
			mesh_type const & geo, _Field<Expression<TOP, T...> > const &expr,
			id_type const & s, traits::iform_list_t<T...>)
	{
		return _invoke_helper(geo, expr, s, typename make_index_sequence<sizeof...(T)>::type ()) ;
			}

//	template<typename TOP, typename TL>
//	static traits::value_type_t<_Field<Expression<TOP, TL> > > eval(
//			mesh_type const & geo, _Field<Expression<TOP, TL>> const &f,
//			id_type const & s, traits::iform_list_t<TL>)
//	{
//		traits::value_type_t<_Field<Expression<TOP, TL> > > res;
//		res = f.m_op_(eval(geo, std::get<0>(f.args), s));
//		return std::move(res);
//	}
//
//	template<typename TOP, typename TL, typename TR>
//	static traits::value_type_t<_Field<Expression<TOP, TL, TR> > > eval(
//			mesh_type const & geo, _Field<Expression<TOP, TL, TR> > const &f,
//			id_type const &s, traits::iform_list_t<TL, TR>)
//	{
//
//		traits::value_type_t<_Field<Expression<TOP, TL, TR> > > res;
//
//		res = (f.m_op_(eval(geo, std::get<0>(f.args), s),
//				eval(geo, std::get<1>(f.args), s)));
//
//		return std::move(res);
//	}

	//***************************************************************************************************
	// Exterior algebra
	//***************************************************************************************************

	template<typename T>
	static inline traits::value_type_t<
			_Field<Expression<tags::ExteriorDerivative, T> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::ExteriorDerivative, T> > const &f,
			id_type s, integer_sequence<int, VERTEX>)
	{
		id_type D = mesh_type::delta_index(s);
		return (eval(geo, std::get<0>(f.args), s + D) * geo.volume(s + D)
				- eval(geo, std::get<0>(f.args), s - D) * geo.volume(s - D))
				* geo.inv_volume(s);
	}

	template<typename T>
	static inline traits::value_type_t<
			_Field<Expression<tags::ExteriorDerivative, T> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::ExteriorDerivative, T> > const & expr,
			id_type s, integer_sequence<int, EDGE>)
	{

		id_type X = mesh_type::delta_index(mesh_type::dual(s));
		id_type Y = mesh_type::rotate(X);
		id_type Z = mesh_type::inverse_rotate(X);

		return ((eval(geo, std::get<0>(expr.args), s + Y) * geo.volume(s + Y) //
		- eval(geo, std::get<0>(expr.args), s - Y) * geo.volume(s - Y))
				- (eval(geo, std::get<0>(expr.args), s + Z) * geo.volume(s + Z) //
				- eval(geo, std::get<0>(expr.args), s - Z) * geo.volume(s - Z) //
				)

		) * geo.inv_volume(s);

	}

	template<typename T>
	static constexpr inline traits::value_type_t<
			_Field<Expression<tags::ExteriorDerivative, T> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::ExteriorDerivative, T> > const & expr,
			id_type s, integer_sequence<int, FACE>)
	{
		return (eval(geo, std::get<0>(expr.args), s + mesh_type::_DI)
				* geo.volume(s + mesh_type::_DI)
				- eval(geo, std::get<0>(expr.args), s - mesh_type::_DI)
						* geo.volume(s - mesh_type::_DI)
				+ eval(geo, std::get<0>(expr.args), s + mesh_type::_DJ)
						* geo.volume(s + mesh_type::_DJ)
				- eval(geo, std::get<0>(expr.args), s - mesh_type::_DJ)
						* geo.volume(s - mesh_type::_DJ)
				+ eval(geo, std::get<0>(expr.args), s + mesh_type::_DK)
						* geo.volume(s + mesh_type::_DK)
				- eval(geo, std::get<0>(expr.args), s - mesh_type::_DK)
						* geo.volume(s - mesh_type::_DK)

		) * geo.inv_volume(s)

		;
	}
//
////	template<typename geometry_type,typename TM, int IL, typename TL> void eval(
////			tags::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
////					typename geometry_type::id_type   s)  = delete;
////
////	template<typename geometry_type,typename TM, int IL, typename TL> void eval(
////			tags::CodifferentialDerivative,
////			_Field<TL...> const & f, 		typename geometry_type::id_type   s)  = delete;

	template<typename T>
	static constexpr inline traits::value_type_t<
			_Field<Expression<tags::CodifferentialDerivative, T> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::CodifferentialDerivative, T>> const & expr,
			id_type s, integer_sequence<int, EDGE>)
	{
		return -(eval(geo, std::get<0>(expr.args), s + mesh_type::_DI)
				* geo.dual_volume(s + mesh_type::_DI)
				- eval(geo, std::get<0>(expr.args), s - mesh_type::_DI)
						* geo.dual_volume(s - mesh_type::_DI)
				+ eval(geo, std::get<0>(expr.args), s + mesh_type::_DJ)
						* geo.dual_volume(s + mesh_type::_DJ)
				- eval(geo, std::get<0>(expr.args), s - mesh_type::_DJ)
						* geo.dual_volume(s - mesh_type::_DJ)
				+ eval(geo, std::get<0>(expr.args), s + mesh_type::_DK)
						* geo.dual_volume(s + mesh_type::_DK)
				- eval(geo, std::get<0>(expr.args), s - mesh_type::_DK)
						* geo.dual_volume(s - mesh_type::_DK)

		) * geo.inv_dual_volume(s)

		;

	}

	template<typename T>
	static inline traits::value_type_t<
			_Field<Expression<tags::CodifferentialDerivative, T> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::CodifferentialDerivative, T>> const & expr,
			id_type s, integer_sequence<int, FACE>)
	{

		id_type X = mesh_type::delta_index(s);
		id_type Y = mesh_type::rotate(X);
		id_type Z = mesh_type::inverse_rotate(X);

		return

		-((eval(geo, std::get<0>(expr.args), s + Y) * (geo.dual_volume(s + Y))
				- eval(geo, std::get<0>(expr.args), s - Y)
						* (geo.dual_volume(s - Y)))

				- (eval(geo, std::get<0>(expr.args), s + Z)
						* (geo.dual_volume(s + Z))
						- eval(geo, std::get<0>(expr.args), s - Z)
								* (geo.dual_volume(s - Z)))

		) * geo.inv_dual_volume(s)

		;
	}

	template<typename T>
	static inline traits::value_type_t<
			_Field<Expression<tags::CodifferentialDerivative, T> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::CodifferentialDerivative, T> > const & expr,
			id_type s, integer_sequence<int, VOLUME>)
	{
		id_type D = mesh_type::delta_index(mesh_type::dual(s));
		return

		-(

		eval(geo, std::get<0>(expr.args), s + D) * (geo.dual_volume(s + D)) //
		- eval(geo, std::get<0>(expr.args), s - D) * (geo.dual_volume(s - D))

		) * geo.inv_dual_volume(s)

		;
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, VERTEX, VERTEX>)
	{
		return (eval(geo, std::get<0>(expr.args), s)
				* eval(geo, std::get<1>(expr.args), s));
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const& geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, VERTEX, EDGE>)
	{
		auto X = mesh_type::delta_index(s);

		return (eval(geo, std::get<0>(expr.args), s - X)
				+ eval(geo, std::get<0>(expr.args), s + X)) * 0.5
				* eval(geo, std::get<1>(expr.args), s);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, VERTEX, FACE>)
	{
		auto X = mesh_type::delta_index(mesh_type::dual(s));
		auto Y = mesh_type::rotate(X);
		auto Z = mesh_type::inverse_rotate(X);

		return (

		eval(geo, std::get<0>(expr.args), (s - Y) - Z)
				+ eval(geo, std::get<0>(expr.args), (s - Y) + Z)
				+ eval(geo, std::get<0>(expr.args), (s + Y) - Z)
				+ eval(geo, std::get<0>(expr.args), (s + Y) + Z)

		) * 0.25 * eval(geo, std::get<1>(expr.args), s);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, VERTEX, VOLUME>)
	{

		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);

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
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, EDGE, VERTEX>)
	{

		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);

		auto X = mesh_type::delta_index(s);
		return eval(geo, l, s) * (eval(geo, r, s - X) + eval(geo, r, s + X))
				* 0.5;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, EDGE, EDGE>)
	{
		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);

		auto Y = mesh_type::delta_index(mesh_type::rotate(mesh_type::dual(s)));
		auto Z = mesh_type::delta_index(
				mesh_type::inverse_rotate(mesh_type::dual(s)));

		return ((eval(geo, l, s - Y) + eval(geo, l, s + Y))
				* (eval(geo, l, s - Z) + eval(geo, l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, EDGE, FACE>)
	{
		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

		(eval(geo, l, (s - Y) - Z) + eval(geo, l, (s - Y) + Z)
				+ eval(geo, l, (s + Y) - Z) + eval(geo, l, (s + Y) + Z))
				* (eval(geo, r, s - X) + eval(geo, r, s + X))
				+

				(eval(geo, l, (s - Z) - X) + eval(geo, l, (s - Z) + X)
						+ eval(geo, l, (s + Z) - X) + eval(geo, l, (s + Z) + X))
						* (eval(geo, r, s - Y) + eval(geo, r, s + Y))
				+

				(eval(geo, l, (s - X) - Y) + eval(geo, l, (s - X) + Y)
						+ eval(geo, l, (s + X) - Y) + eval(geo, l, (s + X) + Y))
						* (eval(geo, r, s - Z) + eval(geo, r, s + Z))

		) * 0.125;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR> > const & expr, id_type s,
			integer_sequence<int, FACE, VERTEX>)
	{
		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);
		auto Y = mesh_type::delta_index(mesh_type::rotate(mesh_type::dual(s)));
		auto Z = mesh_type::delta_index(
				mesh_type::inverse_rotate(mesh_type::dual(s)));

		return eval(geo, l, s)
				* (eval(geo, r, (s - Y) - Z) + eval(geo, r, (s - Y) + Z)
						+ eval(geo, r, (s + Y) - Z) + eval(geo, r, (s + Y) + Z))
				* 0.25;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR> > const & expr, id_type s,
			integer_sequence<int, FACE, EDGE>)
	{
		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

		(eval(geo, r, (s - Y) - Z) + eval(geo, r, (s - Y) + Z)
				+ eval(geo, r, (s + Y) - Z) + eval(geo, r, (s + Y) + Z))
				* (eval(geo, l, s - X) + eval(geo, l, s + X))

				+ (eval(geo, r, (s - Z) - X) + eval(geo, r, (s - Z) + X)
						+ eval(geo, r, (s + Z) - X) + eval(geo, r, (s + Z) + X))
						* (eval(geo, l, s - Y) + eval(geo, l, s + Y))

				+ (eval(geo, r, (s - X) - Y) + eval(geo, r, (s - X) + Y)
						+ eval(geo, r, (s + X) - Y) + eval(geo, r, (s + X) + Y))
						* (eval(geo, l, s - Z) + eval(geo, l, s + Z))

		) * 0.125;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<_Field<Expression<tags::Wedge, TL, TR> > > eval(
			mesh_type const & geo,
			_Field<Expression<tags::Wedge, TL, TR>> const & expr, id_type s,
			integer_sequence<int, VOLUME, VERTEX>)
	{
		auto const & l = std::get<0>(expr.args);
		auto const & r = std::get<1>(expr.args);
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		eval(geo, l, s) * (

		eval(geo, r, ((s - X) - Y) - Z) + //
				eval(geo, r, ((s - X) - Y) + Z) + //
				eval(geo, r, ((s - X) + Y) - Z) + //
				eval(geo, r, ((s - X) + Y) + Z) + //
				eval(geo, r, ((s + X) - Y) - Z) + //
				eval(geo, r, ((s + X) - Y) + Z) + //
				eval(geo, r, ((s + X) + Y) - Z) + //
				eval(geo, r, ((s + X) + Y) + Z) //

		) * 0.125;
	}
//
//////***************************************************************************************************
//
//	template<typename T>
//	static inline traits::value_type_t<_Field<Expression<tags::HodgeStar, T> > > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::HodgeStar, T> > const & expr, id_type s,
//			integer_sequence<int, VOLUME>)
//	{
//		auto const & f = std::get<0>(expr.args);
////		auto X = geo.DI(0,s);
////		auto Y = geo.DI(1,s);
////		auto Z =geo.DI(2,s);
////
////		return
////
////		(
////
////		eval(geo,f,((s + X) - Y) - Z)*geo.inv_volume(((s + X) - Y) - Z) +
////
////		eval(geo,f,((s + X) - Y) + Z)*geo.inv_volume(((s + X) - Y) + Z) +
////
////		eval(geo,f,((s + X) + Y) - Z)*geo.inv_volume(((s + X) + Y) - Z) +
////
////		eval(geo,f,((s + X) + Y) + Z)*geo.inv_volume(((s + X) + Y) + Z) +
////
////		eval(geo,f,((s - X) - Y) - Z)*geo.inv_volume(((s - X) - Y) - Z) +
////
////		eval(geo,f,((s - X) - Y) + Z)*geo.inv_volume(((s - X) - Y) + Z) +
////
////		eval(geo,f,((s - X) + Y) - Z)*geo.inv_volume(((s - X) + Y) - Z) +
////
////		eval(geo,f,((s - X) + Y) + Z)*geo.inv_volume(((s - X) + Y) + Z)
////
////		) * 0.125 * geo.volume(s);
//
//		return eval(geo, f, s) /** geo.tags::HodgeStarVolumeScale(s)*/;
//	}
//
////	template<typename geometry_type,typename TM, typename TL, typename TR> void eval(
////			tags::InteriorProduct, nTuple<TR, G::ndims> const & v,
////			_Field<Domain<TM, VERTEX>, TL> const & f,
////					typename geometry_type::id_type   s)  = delete;
//
//	template<typename TL, typename TR>
//	static inline traits::value_type_t<
//			_Field<Expression<tags::InteriorProduct, TL, TR> > > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::InteriorProduct, TL, TR>> const & expr,
//			id_type s, integer_sequence<int, EDGE, VERTEX>)
//	{
//		auto const & f = std::get<0>(expr.args);
//		auto const & v = std::get<1>(expr.args);
//
//		auto X = geo.DI(0, s);
//		auto Y = geo.DI(1, s);
//		auto Z = geo.DI(2, s);
//
//		return
//
//		(eval(geo, f, s + X) - eval(geo, f, s - X)) * 0.5 * v[0] //
//		+ (eval(geo, f, s + Y) - eval(geo, f, s - Y)) * 0.5 * v[1] //
//		+ (eval(geo, f, s + Z) - eval(geo, f, s - Z)) * 0.5 * v[2];
//	}
//
//	template<typename TL, typename TR>
//	static inline traits::value_type_t<
//			_Field<Expression<tags::InteriorProduct, TL, TR> > > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::InteriorProduct, TL, TR>> const & expr,
//			id_type s, integer_sequence<int, FACE, VERTEX>)
//	{
//		auto const & f = std::get<0>(expr.args);
//		auto const & v = std::get<1>(expr.args);
//
//		int n = mesh_type::component_number(s);
//
//		auto X = mesh_type::delta_index(s);
//		auto Y = mesh_type::rotate(X);
//		auto Z = mesh_type::inverse_rotate(X);
//		return
//
//		(eval(geo, f, s + Y) + eval(geo, f, s - Y)) * 0.5 * v[(n + 2) % 3] -
//
//		(eval(geo, f, s + Z) + eval(geo, f, s - Z)) * 0.5 * v[(n + 1) % 3];
//	}
//
//	template<typename TL, typename TR>
//	static inline traits::value_type_t<
//			_Field<Expression<tags::InteriorProduct, TL, TR> > > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::InteriorProduct, TL, TR>> const & expr,
//			id_type s, integer_sequence<int, VOLUME, VERTEX>)
//	{
//		auto const & f = std::get<0>(expr.args);
//		auto const & v = std::get<1>(expr.args);
//		int n = mesh_type::component_number(mesh_type::dual(s));
//		id_type D = mesh_type::delta_index(mesh_type::dual(s));
//
//		return (eval(geo, f, s + D) - eval(geo, f, s - D)) * 0.5 * v[n];
//	}
//
////**************************************************************************************************
//// Non-standard operation
//
//	template<typename T>
//	static inline traits::value_type_t<T> eval(mesh_type const & geo,
//			_Field<Expression<tags::MapTo, typename traits::iform<T>::type, T> > const & f,
//			id_type s, typename traits::iform_list<T>::type)
//	{
//		return eval(geo, f, s);
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,std::integral_constant<int,EDGE>, T>> > //
//	eval( mesh_type const & geo,
//			_Field<Expression<tags::MapTo,integer_sequence<int,EDGE>, T>> const & expr,
//			id_type s,integer_sequence< int, VERTEX >)
//	{
//		auto const & f = std::get<1>(expr.args);
//
//		auto X = geo.DI(0, s);
//		auto Y = geo.DI(1, s);
//		auto Z = geo.DI(2, s);
//
//		return nTuple<
//		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
//		3>(
//				{
//
//					(eval(geo, f, s - X) + eval(geo, f, s + X)) * 0.5, //
//					(eval(geo, f, s - Y) + eval(geo, f, s + Y)) * 0.5,//
//					(eval(geo, f, s - Z) + eval(geo, f, s + Z)) * 0.5
//
//				});
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,std::integral_constant<int,VERTEX>, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo,std::integral_constant<int,VERTEX>, T>> const & expr,
//			id_type s,integer_sequence< int, EDGE >)
//	{
//		auto const & f = std::get<1>(expr.args);
//		auto n = mesh_type::component_number(s);
//		auto D = mesh_type::delta_index(s);
//
//		return ((eval(geo, f, s - D)[n] + eval(geo, f, s + D)[n])
//				* 0.5);
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,std::integral_constant<int,FACE>, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo, std::integral_constant<int,FACE>,T> > const & expr,
//			id_type s,integer_sequence< int, EDGE >)
//	{
//		auto const & f = std::get<0>(expr.args);
//		auto X = geo.DI(0, s);
//		auto Y = geo.DI(1, s);
//		auto Z = geo.DI(2, s);
//
//		return nTuple<
//		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
//		3>(
//				{	(
//
//							eval(geo, f, (s - Y) - Z) +
//
//							eval(geo, f, (s - Y) + Z) +
//
//							eval(geo, f, (s + Y) - Z) +
//
//							eval(geo, f, (s + Y) + Z)
//
//					) * 0.25,
//
//					(
//
//							eval(geo, f, (s - Z) - X) +
//
//							eval(geo, f, (s - Z) + X) +
//
//							eval(geo, f, (s + Z) - X) +
//
//							eval(geo, f, (s + Z) + X)
//
//					) * 0.25,
//
//					(
//
//							eval(geo, f, (s - X) - Y) +
//
//							eval(geo, f, (s - X) + Y) +
//
//							eval(geo, f, (s + X) - Y) +
//
//							eval(geo, f, (s + X) + Y)
//
//					) * 0.25
//
//				});
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,std::integral_constant<int,VERTEX>, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo,std::integral_constant<int,VERTEX>, T>> const & expr,
//			id_type s,integer_sequence<int, FACE >)
//	{
//		auto const & f = std::get<0>(expr.args);
//
//		auto n = mesh_type::component_number(mesh_type::dual(s));
//		auto X = mesh_type::delta_index(mesh_type::dual(s));
//		auto Y = mesh_type::rotate(X);
//		auto Z = mesh_type::inverse_rotate(X);
//
//		return (
//
//				(
//
//						eval(geo, f, (s - Y) - Z)[n] +
//
//						eval(geo, f, (s - Y) + Z)[n] +
//
//						eval(geo, f, (s + Y) - Z)[n] +
//
//						eval(geo, f, (s + Y) + Z)[n]
//
//				) * 0.25
//
//		);
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,FACE, VOLUME, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo,FACE, VOLUME, T>> const & expr,
//			id_type s)
//	{
//		auto const & f = std::get<0>(expr.args);
//
//		auto X = geo.DI(0, s);
//		auto Y = geo.DI(1, s);
//		auto Z = geo.DI(2, s);
//
//		return nTuple<
//		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
//		3>(
//				{
//
//					(eval(geo, f, s - X) + eval(geo, f, s + X)) * 0.5, //
//					(eval(geo, f, s - Y) + eval(geo, f, s + Y)) * 0.5,//
//					(eval(geo, f, s - Z) + eval(geo, f, s + Z)) * 0.5
//
//				});
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,VOLUME, FACE, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo,VOLUME, FACE, T>> const & expr,
//			id_type s)
//	{
//		auto const & f = std::get<0>(expr.args);
//
//		auto n = mesh_type::component_number(mesh_type::dual(s));
//		auto D = mesh_type::delta_index(mesh_type::dual(s));
//
//		return ((eval(geo, f, s - D)[n] + eval(geo, f, s + D)[n])
//				* 0.5);
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,EDGE, VOLUME, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo,EDGE, VOLUME, T>> const & expr,
//			id_type s)
//	{
//		auto const & f = std::get<0>(expr.args);
//
//		auto X = geo.DI(0, s);
//		auto Y = geo.DI(1, s);
//		auto Z = geo.DI(2, s);
//
//		return nTuple<
//		typename std::remove_reference<decltype(eval(geo,f,s))>::type,
//		3>(
//				{	(
//
//							eval(geo, f, (s - Y) - Z) +
//
//							eval(geo, f, (s - Y) + Z) +
//
//							eval(geo, f, (s + Y) - Z) +
//
//							eval(geo, f, (s + Y) + Z)
//
//					) * 0.25,
//
//					(
//
//							eval(geo, f, (s - Z) - X) +
//
//							eval(geo, f, (s - Z) + X) +
//
//							eval(geo, f, (s + Z) - X) +
//
//							eval(geo, f, (s + Z) + X)
//
//					) * 0.25,
//
//					(
//
//							eval(geo, f, (s - X) - Y) +
//
//							eval(geo, f, (s - X) + Y) +
//
//							eval(geo, f, (s + X) - Y) +
//
//							eval(geo, f, (s + X) + Y)
//
//					) * 0.25,
//
//				});
//	}
//
//	template<typename T>
//	traits::value_type_t<_Field<Expression<tags::MapTo,VOLUME, EDGE, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::MapTo,VOLUME, EDGE, T>> const & expr,
//			id_type s)
//	{
//		auto const & f = std::get<0>(expr.args);
//		auto n = mesh_type::component_number(mesh_type::dual(s));
//		auto X = mesh_type::delta_index(mesh_type::dual(s));
//		auto Y = mesh_type::rotate(X);
//		auto Z = mesh_type::inverse_rotate(X);
//		return (
//
//				(
//
//						eval(geo, f, (s - Y) - Z)[n] +
//
//						eval(geo, f, (s - Y) + Z)[n] +
//
//						eval(geo, f, (s + Y) - Z)[n] +
//
//						eval(geo, f, (s + Y) + Z)[n]
//
//				) * 0.25
//
//		);
//	}
//
//	// For curl_pdx
//
//	template<int N, typename T> static inline traits::value_type_t<
//	_Field<Expression<tags::ExteriorDerivative,N, EDGE, T> > > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::ExteriorDerivative,N, EDGE, T>> const & expr,
//			id_type s)
//	{
//		auto const & f = std::get<0>(expr.args);
//
//		auto X = mesh_type::delta_index(mesh_type::dual(s));
//		auto Y = mesh_type::rotate(X);
//		auto Z = mesh_type::inverse_rotate(X);
//
//		Y = (mesh_type::component_number(Y) == N) ? Y : 0UL;
//		Z = (mesh_type::component_number(Z) == N) ? Z : 0UL;
//
//		return (eval(geo, f, s + Y) - eval(geo, f, s - Y))
//		- (eval(geo, f, s + Z) - eval(geo, f, s - Z));
//	}
//
//	template<int N, typename T> static inline traits::value_type_t<
//	_Field<Expression<tags::CodifferentialDerivative,N, FACE, T>> > eval(
//			mesh_type const & geo,
//			_Field<Expression<tags::CodifferentialDerivative,N, FACE, T>> const & expr,
//			id_type s)
//	{
//		auto const & f = std::get<0>(expr.args);
//
//		auto X = mesh_type::delta_index(s);
//		auto Y = mesh_type::rotate(X);
//		auto Z = mesh_type::inverse_rotate(X);
//
//		Y = (mesh_type::component_number(Y) == N) ? Y : 0UL;
//		Z = (mesh_type::component_number(Z) == N) ? Z : 0UL;
//
//		return (
//
//				eval(geo, f, s + Y) * (geo.dual_volume(s + Y))      //
//				- eval(geo, f, s - Y) * (geo.dual_volume(s - Y))//
//				- eval(geo, f, s + Z) * (geo.dual_volume(s + Z))//
//				+ eval(geo, f, s - Z) * (geo.dual_volume(s - Z))//
//
//		) * geo.inv_dual_volume(s);
//	}

};
}
// namespace policy
				}
// namespace simpla

#endif /* FDM_H_ */
