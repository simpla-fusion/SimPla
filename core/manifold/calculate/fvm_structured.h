/**
 * @file  calculate_fvm.h
 *
 *  Created on: 2014-9-23
 *      Author: salmon
 */

#ifndef CALCULATE_FVM_H_
#define CALCULATE_FVM_H_

#include <complex>
#include <cstddef>
#include <type_traits>

#include "calculate.h"

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/type_traits.h"
#include "../../manifold/manifold_traits.h"

namespace simpla
{

template<typename ...> class Field;

/** @ingroup diff_scheme
 *  @brief   FdMesh
 */

namespace calculate
{

template<typename TGeo> using FVM=  Calculate<TGeo, tags::finite_volume>;


template<typename Geometry>
struct Calculate<Geometry, tags::finite_volume>
{
private:

	typedef Geometry mesh_type;

	typedef Calculate<mesh_type, tags::finite_volume> this_type;

//	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::id_type id;

public:
	///***************************************************************************************************
	/// @name general_algebra General algebra
	/// @{
	///***************************************************************************************************

	static constexpr Real eval(mesh_type const &geo, Real v, id s)
	{
		return v;
	}

	static constexpr int eval(mesh_type const &geo, int v, id s)
	{
		return v;
	}

	static constexpr std::complex<Real> eval(mesh_type const &geo,
			std::complex<Real> v, id s)
	{
		return v;
	}

	template<typename T, size_t ...N>
	static constexpr nTuple<T, N...> const &eval(mesh_type const &geo,
			nTuple<T, N...> const &v, id s)
	{

		return v;
	}

	template<typename ...T>
	static inline traits::primary_type_t<nTuple<Expression<T...>>>
	eval(
			mesh_type const
			&geo,
			nTuple<Expression<T...>> const &v,
			id
			s)
	{
		traits::primary_type_t<nTuple<Expression<T...> > > res;
		res = v;
		return std::move(res);
	}

	template<typename TM, typename TV, typename ... Others, typename ... Args>
	static constexpr TV eval(mesh_type const &geo, Field<TM, TV, Others...> const &f, id s)
	{
		return traits::index(f, s);
	}

	template<typename TOP, typename ... T>
	static constexpr traits::primary_type_t<
			traits::value_type_t<Field<Expression<TOP, T...> >>>
	eval(
			mesh_type const
			&geo, Field<Expression<TOP, T...> > const &expr,
			id const &s
	)
	{
		return eval(geo, expr, s, traits::iform_list_t<T...>());
	}

private:

	template<typename Expr, size_t ... index>
	static traits::primary_type_t<traits::value_type_t<Expr>> _invoke_helper(
			mesh_type const &geo, Expr const &expr, id s,
			index_sequence<index...>)
	{
		traits::primary_type_t<traits::value_type_t<Expr>> res = (expr.m_op_(
				eval(geo, std::get<index>(expr.args), s)...));

		return std::move(res);
	}

public:

	template<typename TOP, typename ... T>
	static constexpr traits::primary_type_t<
			traits::value_type_t<Field<Expression<TOP, T...> >>>
	eval(
			mesh_type const
			&geo, Field<Expression<TOP, T...> > const &expr,
			id const &s, traits::iform_list_t<T...>
	)
	{
		return _invoke_helper(geo, expr, s, typename make_index_sequence<sizeof...(T)>::type());
	}



	//***************************************************************************************************
	// Exterior algebra
	//***************************************************************************************************

	template<typename T>
	static inline traits::value_type_t<
			Field<Expression<tags::ExteriorDerivative, T>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::ExteriorDerivative, T> > const &f,
			id s, integer_sequence<int, VERTEX>
	)
	{
		id D = mesh_type::delta_index(s);
		return (eval(geo, std::get<0>(f.args), s + D) * geo.volume(s + D)
				- eval(geo, std::get<0>(f.args), s - D) * geo.volume(s - D))
				* geo.inv_volume(s);
	}

	template<typename T>
	static inline traits::value_type_t<
			Field<Expression<tags::ExteriorDerivative, T>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::ExteriorDerivative, T> > const &expr,
			id s, integer_sequence<int, EDGE>
	)
	{

		id X = mesh_type::delta_index(mesh_type::dual(s));
		id Y = mesh_type::rotate(X);
		id Z = mesh_type::inverse_rotate(X);

		return ((eval(geo, std::get<0>(expr.args), s + Y) * geo.volume(s + Y) //
				- eval(geo, std::get<0>(expr.args), s - Y) * geo.volume(s - Y))
				- (eval(geo, std::get<0>(expr.args), s + Z) * geo.volume(s + Z) //
				- eval(geo, std::get<0>(expr.args), s - Z) * geo.volume(s - Z) //
		)

		) * geo.inv_volume(s);

	}

	template<typename T>
	static constexpr inline traits::value_type_t<
			Field<Expression<tags::ExteriorDerivative, T>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::ExteriorDerivative, T> > const &expr,
			id s, integer_sequence<int, FACE>
	)
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

		) * geo.inv_volume(s);
	}
//
////	template<typename geometry_type,typename TM, int IL, typename TL> void eval(
////			tags::ExteriorDerivative, Field<Domain<TM, IL>, TL> const & f,
////					typename geometry_type::id_type   s)  = delete;
////
////	template<typename geometry_type,typename TM, int IL, typename TL> void eval(
////			tags::CodifferentialDerivative,
////			Field<TL...> const & f, 		typename geometry_type::id_type   s)  = delete;

	template<typename T>
	static constexpr inline traits::value_type_t<
			Field<Expression<tags::CodifferentialDerivative, T>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::CodifferentialDerivative, T>> const &expr,
			id s, integer_sequence<int, EDGE>
	)
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

		) * geo.inv_dual_volume(s);

	}

	template<typename T>
	static inline traits::value_type_t<
			Field<Expression<tags::CodifferentialDerivative, T>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::CodifferentialDerivative, T>> const &expr,
			id s, integer_sequence<int, FACE>
	)
	{

		id X = mesh_type::delta_index(s);
		id Y = mesh_type::rotate(X);
		id Z = mesh_type::inverse_rotate(X);

		return

				-((eval(geo, std::get<0>(expr.args), s + Y) * (geo.dual_volume(s + Y))
						- eval(geo, std::get<0>(expr.args), s - Y)
						* (geo.dual_volume(s - Y)))

						- (eval(geo, std::get<0>(expr.args), s + Z)
						* (geo.dual_volume(s + Z))
						- eval(geo, std::get<0>(expr.args), s - Z)
						* (geo.dual_volume(s - Z)))

				) * geo.inv_dual_volume(s);
	}

	template<typename T>
	static inline traits::value_type_t<
			Field<Expression<tags::CodifferentialDerivative, T>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::CodifferentialDerivative, T> > const &expr,
			id s, integer_sequence<int, VOLUME>
	)
	{
		id D = mesh_type::delta_index(mesh_type::dual(s));
		return

				-(

						eval(geo, std::get<0>(expr.args), s + D) * (geo.dual_volume(s + D)) //
								- eval(geo, std::get<0>(expr.args), s - D) * (geo.dual_volume(s - D))

				) * geo.inv_dual_volume(s);
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL, typename TR>
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, VERTEX, VERTEX>
	)
	{
		return (eval(geo, std::get<0>(expr.args), s)
				* eval(geo, std::get<1>(expr.args), s));
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, VERTEX, EDGE>
	)
	{
		auto X = mesh_type::delta_index(s);

		return (eval(geo, std::get<0>(expr.args), s - X)
				+ eval(geo, std::get<0>(expr.args), s + X)) * 0.5
				* eval(geo, std::get<1>(expr.args), s);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, VERTEX, FACE>
	)
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
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, VERTEX, VOLUME>
	)
	{

		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);

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
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, EDGE, VERTEX>
	)
	{

		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);

		auto X = mesh_type::delta_index(s);
		return eval(geo, l, s) * (eval(geo, r, s - X) + eval(geo, r, s + X))
				* 0.5;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, EDGE, EDGE>
	)
	{
		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);

		auto Y = mesh_type::delta_index(mesh_type::rotate(mesh_type::dual(s)));
		auto Z = mesh_type::delta_index(
				mesh_type::inverse_rotate(mesh_type::dual(s)));

		return ((eval(geo, l, s - Y) + eval(geo, l, s + Y))
				* (eval(geo, l, s - Z) + eval(geo, l, s + Z)) * 0.25);
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, EDGE, FACE>
	)
	{
		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);
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
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR> > const &expr,
			id s,
			integer_sequence<int, FACE, VERTEX>
	)
	{
		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);
		auto Y = mesh_type::delta_index(mesh_type::rotate(mesh_type::dual(s)));
		auto Z = mesh_type::delta_index(
				mesh_type::inverse_rotate(mesh_type::dual(s)));

		return eval(geo, l, s)
				* (eval(geo, r, (s - Y) - Z) + eval(geo, r, (s - Y) + Z)
				+ eval(geo, r, (s + Y) - Z) + eval(geo, r, (s + Y) + Z))
				* 0.25;
	}

	template<typename TL, typename TR>
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(
			mesh_type const
			&geo,
			Field<Expression<tags::Wedge, TL, TR> > const &expr,
			id s,
			integer_sequence<int, FACE, EDGE>
	)
	{
		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);
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
	static inline traits::value_type_t<Field<Expression<tags::Wedge, TL, TR>>>
	eval(mesh_type const
	&geo, Field<Expression<tags::Wedge, TL, TR>> const &expr,
			id s,
			integer_sequence<int, VOLUME, VERTEX>
	)
	{
		auto const &l = std::get<0>(expr.args);
		auto const &r = std::get<1>(expr.args);
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
};// struct Calculate<Geometry, tags::finite_volume>
}// namespace calculate
}// namespace simpla

#endif /* FDM_H_ */
