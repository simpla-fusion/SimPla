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

#include "../../gtl/expression_template.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/primitives.h"
#include "../calculus.h"
#include "../mesh.h"

namespace simpla
{

template<typename ... > class _Field;
template<typename ...> class Domain;
template<typename ...>class field_traits;

/** @ingroup diff_scheme
 *  @brief   FdMesh
 */
template<typename G>
struct FiniteDiffMethod
{
public:

	typedef FiniteDiffMethod<G> this_type;
	typedef G geometry_type;
	typedef typename geometry_type::topology_type topology_type;
	typedef typename topology_type::id_type id_type;
	typedef Real scalar_type;
	static constexpr size_t NUM_OF_COMPONENT_TYPE = G::ndims + 1;
	static constexpr size_t ndims = G::ndims;

	FiniteDiffMethod()
	{
	}

	FiniteDiffMethod(this_type const & r) = default;

	~FiniteDiffMethod() = default;

	this_type & operator=(this_type const &) = default;
	//***************************************************************************************************
	// General algebra
	//***************************************************************************************************

	template<typename ...Others>
	static Real calculate(geometry_type const & geo, Real v, Others &&... s)
	{
		return v;
	}

	template<typename ...Others>
	static int calculate(geometry_type const & geo, int v, Others &&... s)
	{
		return v;
	}

	template<typename ...Others>
	static std::complex<Real> calculate(geometry_type const & geo,
			std::complex<Real> v, Others &&... s)
	{
		return v;
	}

	template<typename T, size_t ...N, typename ...Others>
	static nTuple<T, N...> const& calculate(geometry_type const & geo,
			nTuple<T, N...> const& v, Others &&... s)
	{
		return v;
	}

	template<typename ...T, typename ...Others>
	static inline typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type
	calculate(geometry_type const & geo,nTuple<Expression<T...>> const & v, Others &&... s)
	{
		typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type res;
		res=v;
		return std::move(res);
	}

	template<typename TM,typename TV, typename ... Others,typename ... Args>
	static inline TV
	calculate(geometry_type const & geo,_Field<TM, TV, Others...> const &f, Args && ... s)
	{
		return get_value(f,std::forward<Args>(s)...);
	}

	template<typename TOP, typename TL, typename TR, typename ...Others>
	static inline typename field_traits< _Field<Expression<TOP, TL, TR>>>::value_type
	calculate(geometry_type const & geo,_Field<Expression<TOP, TL, TR>> const &f, Others &&... s)
	{
		return f.op_(calculate( geo,f.lhs,std::forward<Others>(s)...),
				calculate(geo,f.rhs,std::forward<Others>(s)...));
	}

	template<typename TOP, typename TL, typename ...Others>
	static inline typename field_traits< _Field<Expression<TOP, TL,std::nullptr_t>>>::value_type
	calculate(geometry_type const & geo,_Field<Expression<TOP, TL,std::nullptr_t>> const &f, Others &&... s)
	{
		return f.op_(calculate(geo,f.lhs,std::forward<Others>(s)...) );
	}

	//***************************************************************************************************
	// Exterior algebra
	//***************************************************************************************************

	template<typename T>
	static inline typename field_traits<_Field<_impl::ExteriorDerivative<VERTEX,T> >>::value_type
	calculate(geometry_type const & geo,_Field<_impl::ExteriorDerivative<VERTEX,T> > const & f, id_type s)
	{
		constexpr id_type D = topology_type::delta_index(s);

		return (calculate(geo,f.lhs, s + D) * geo.volume(s + D)
				- calculate(geo,f.lhs, s - D) * geo.volume(s - D)) * geo.inv_volume(s);
	}

	template<typename T>
	static inline typename field_traits<_Field<_impl::ExteriorDerivative< EDGE,T> >>::value_type
	calculate(geometry_type const & geo,_Field<_impl::ExteriorDerivative<EDGE,T> > const & expr, id_type s)
	{

		id_type X = topology_type::delta_index(topology_type::dual(s));
		id_type Y = topology_type::roate(X);
		id_type Z = topology_type::inverse_roate(X);

		return (
				(
						calculate(geo,expr.lhs, s + Y) * geo.volume(s + Y) //
						- calculate(geo,expr.lhs, s - Y) * geo.volume(s - Y) ) - (
						calculate(geo,expr.lhs, s + Z) * geo.volume(s + Z)//
						- calculate(geo,expr.lhs, s - Z) * geo.volume(s - Z)//
				)

		) * geo.inv_volume(s);

	}

	template<typename T>
	static constexpr inline typename field_traits<_Field<_impl::ExteriorDerivative< FACE,T> >>::value_type
	calculate(geometry_type const & geo,_Field<_impl::ExteriorDerivative<FACE,T> > const & expr, id_type s)
	{
		return (
				calculate(geo,expr.lhs, s + topology_type::_DI) * geo.volume(s + topology_type::_DI)
				- calculate(geo,expr.lhs, s - topology_type::_DI) * geo.volume(s - topology_type::_DI)
				+ calculate(geo,expr.lhs, s + topology_type::_DJ) * geo.volume(s + topology_type::_DJ)
				- calculate(geo,expr.lhs, s - topology_type::_DJ) * geo.volume(s - topology_type::_DJ)
				+ calculate(geo,expr.lhs, s + topology_type::_DK) * geo.volume(s + topology_type::_DK)
				- calculate(geo,expr.lhs, s - topology_type::_DK) * geo.volume(s - topology_type::_DK)

		) * geo.inv_volume(s)

		;
	}
//
////	template<typename TM, size_t IL, typename TL> void calculate(
////			_impl::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
////			id_type   s)  = delete;
////
////	template<typename TM, size_t IL, typename TL> void calculate(
////			_impl::CodifferentialDerivative,
////			_Field<TL...> const & f, id_type   s)  = delete;

	template<typename T >
	static constexpr inline typename field_traits<_Field< _impl::CodifferentialDerivative< EDGE, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::CodifferentialDerivative< EDGE, T>> const & expr,
			id_type s)
	{
		return
		-(
				calculate(geo,expr.lhs, s + topology_type::_DI) * geo.dual_volume(s + topology_type::_DI)
				- calculate(geo,expr.lhs, s - topology_type::_DI) * geo.dual_volume(s - topology_type::_DI)
				+ calculate(geo,expr.lhs, s + topology_type::_DJ) * geo.dual_volume(s + topology_type::_DJ)
				- calculate(geo,expr.lhs, s - topology_type::_DJ) * geo.dual_volume(s - topology_type::_DJ)
				+ calculate(geo,expr.lhs, s + topology_type::_DK) * geo.dual_volume(s + topology_type::_DK)
				- calculate(geo,expr.lhs, s - topology_type::_DK) * geo.dual_volume(s - topology_type::_DK)

		) * geo.inv_dual_volume(s)

		;

	}

	template<typename T >
	static inline typename field_traits<_Field< _impl::CodifferentialDerivative< FACE, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::CodifferentialDerivative< FACE, T>> const & expr,
			id_type s)
	{

		id_type X = topology_type::delta_index(s);
		id_type Y = topology_type::roate(X);
		id_type Z = topology_type::inverse_roate(X);

		return

		-(
				(calculate(geo,expr.lhs, s + Y) * (geo.dual_volume(s + Y))
						- calculate(geo,expr.lhs, s - Y) * (geo.dual_volume(s - Y)))

				- (calculate(geo,expr.lhs, s + Z) * (geo.dual_volume(s + Z))
						- calculate(geo,expr.lhs, s - Z) * (geo.dual_volume(s - Z)))

		) * geo.inv_dual_volume(s)

		;
	}

	template<typename T >
	static inline typename field_traits<_Field< _impl::CodifferentialDerivative< VOLUME, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::CodifferentialDerivative< VOLUME, T>> const & expr,
			id_type s)
	{
		id_type D = topology_type::delta_index(topology_type::dual(s));
		return

		-(

				calculate(geo,expr.lhs, s + D) * (geo.dual_volume(s + D)) //
				- calculate(geo,expr.lhs, s - D) * (geo.dual_volume(s - D))

		) * geo.inv_dual_volume(s)

		;
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<VERTEX,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,VERTEX,TL,TR>> const & expr,
			id_type s)
	{
		return (calculate(expr.lhs, s) * calculate(expr.rhs, s));
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<VERTEX,EDGE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,EDGE,TL,TR>> const & expr,
			id_type s)
	{
		auto X = topology_type::delta_index(s);

		return (calculate(expr.lhs, s - X) + calculate(expr.lhs, s + X)) * 0.5
		* calculate(expr.rhs, s);
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<VERTEX,FACE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,FACE,TL,TR>> const & expr,
			id_type s)
	{
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

				calculate(geo,expr.lhs, (s - Y) - Z) +
				calculate(geo,expr.lhs, (s - Y) + Z) +
				calculate(geo,expr.lhs, (s + Y) - Z) +
				calculate(geo,expr.lhs, (s + Y) + Z)

		) * 0.25 * calculate(geo,expr.rhs, s);
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<VERTEX,VOLUME,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,VOLUME,TL,TR>> const & expr,
			id_type s)
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return (

				calculate(geo,l, ((s - X) - Y) - Z) +

				calculate(geo,l, ((s - X) - Y) + Z) +

				calculate(geo,l, ((s - X) + Y) - Z) +

				calculate(geo,l, ((s - X) + Y) + Z) +

				calculate(geo,l, ((s + X) - Y) - Z) +

				calculate(geo,l, ((s + X) - Y) + Z) +

				calculate(geo,l, ((s + X) + Y) - Z) +

				calculate(geo,l, ((s + X) + Y) + Z)

		) * 0.125 * calculate(geo,r, s);
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<EDGE,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<EDGE,VERTEX,TL,TR>> const & expr,
			id_type s)
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = topology_type::delta_index(s);
		return calculate(geo,l, s) * (calculate(geo,r, s - X) + calculate(geo,r, s + X))
		* 0.5;
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<EDGE,EDGE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<EDGE,EDGE,TL,TR>> const & expr,
			id_type s)
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto Y = topology_type::delta_index(topology_type::roate(topology_type::dual(s)));
		auto Z = topology_type::delta_index(topology_type::inverse_roate(topology_type::dual(s)));

		return ((calculate(geo,l, s - Y) + calculate(geo,l, s + Y))
				* (calculate(geo,l, s - Z) + calculate(geo,l, s + Z)) * 0.25);
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<EDGE,FACE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<EDGE,FACE,TL,TR>> const & expr,
			id_type s)
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

				(calculate(geo,l, (s - Y) - Z)
						+ calculate(geo,l, (s - Y) + Z)
						+ calculate(geo,l, (s + Y) - Z)
						+ calculate(geo,l, (s + Y) + Z))
				* (calculate(geo,r, s - X) + calculate(geo,r, s + X))
				+

				(calculate(geo,l, (s - Z) - X)
						+ calculate(geo,l, (s - Z) + X)
						+ calculate(geo,l, (s + Z) - X)
						+ calculate(geo,l, (s + Z) + X))
				* (calculate(geo,r, s - Y)
						+ calculate(geo,r, s + Y))
				+

				(calculate(geo,l, (s - X) - Y)
						+ calculate(geo,l, (s - X) + Y)
						+ calculate(geo,l, (s + X) - Y)
						+ calculate(geo,l, (s + X) + Y))
				* (calculate(geo,r, s - Z)
						+ calculate(geo,r, s + Z))

		) * 0.125;
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<FACE,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<FACE,VERTEX,TL,TR>> const & expr,
			id_type s)
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto Y = topology_type::delta_index(topology_type::roate(topology_type::dual(s)));
		auto Z = topology_type::delta_index(topology_type::inverse_roate(topology_type::dual(s)));

		return calculate(geo,l, s)
		* (calculate(geo,r, (s - Y) - Z) + calculate(geo,r, (s - Y) + Z)
				+ calculate(geo,r, (s + Y) - Z)
				+ calculate(geo,r, (s + Y) + Z)) * 0.25;
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<FACE,EDGE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<FACE,EDGE,TL,TR>> const & expr,
			id_type s)
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

				(calculate(geo,r, (s - Y) - Z)
						+ calculate(geo,r, (s - Y) + Z)
						+ calculate(geo,r, (s + Y) - Z)
						+ calculate(geo,r, (s + Y) + Z))
				* (calculate(geo,l, s - X) + calculate(geo,l, s + X))

				+ (calculate(geo,r, (s - Z) - X)
						+ calculate(geo,r, (s - Z) + X)
						+ calculate(geo,r, (s + Z) - X)
						+ calculate(geo,r, (s + Z) + X))
				* (calculate(geo,l, s - Y)
						+ calculate(geo,l, s + Y))

				+ (calculate(geo,r, (s - X) - Y)
						+ calculate(geo,r, (s - X) + Y)
						+ calculate(geo,r, (s + X) - Y)
						+ calculate(geo,r, (s + X) + Y))
				* (calculate(geo,l, s - Z)
						+ calculate(geo,l, s + Z))

		) * 0.125;
	}

	template<typename TL,typename TR>
	static inline typename field_traits<_Field<_impl::Wedge<VOLUME,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VOLUME,VERTEX,TL,TR>> const & expr,
			id_type s)
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		calculate(geo,l, s) * (

				calculate(geo,r, ((s - X) - Y) - Z) + //
				calculate(geo,r, ((s - X) - Y) + Z) +//
				calculate(geo,r, ((s - X) + Y) - Z) +//
				calculate(geo,r, ((s - X) + Y) + Z) +//
				calculate(geo,r, ((s + X) - Y) - Z) +//
				calculate(geo,r, ((s + X) - Y) + Z) +//
				calculate(geo,r, ((s + X) + Y) - Z) +//
				calculate(geo,r, ((s + X) + Y) + Z)//

		) * 0.125;
	}
//
////***************************************************************************************************

	template<typename T >
	static inline typename field_traits<_Field< _impl::HodgeStar< VOLUME, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::HodgeStar< VOLUME, T>> const & expr,
			id_type s)
	{
		auto const & f =expr.lhs;
//		auto X = geo.DI(0,s);
//		auto Y = geo.DI(1,s);
//		auto Z =geo.DI(2,s);
//
//		return
//
//		(
//
//		calculate(geo,f,((s + X) - Y) - Z)*geo.inv_volume(((s + X) - Y) - Z) +
//
//		calculate(geo,f,((s + X) - Y) + Z)*geo.inv_volume(((s + X) - Y) + Z) +
//
//		calculate(geo,f,((s + X) + Y) - Z)*geo.inv_volume(((s + X) + Y) - Z) +
//
//		calculate(geo,f,((s + X) + Y) + Z)*geo.inv_volume(((s + X) + Y) + Z) +
//
//		calculate(geo,f,((s - X) - Y) - Z)*geo.inv_volume(((s - X) - Y) - Z) +
//
//		calculate(geo,f,((s - X) - Y) + Z)*geo.inv_volume(((s - X) - Y) + Z) +
//
//		calculate(geo,f,((s - X) + Y) - Z)*geo.inv_volume(((s - X) + Y) - Z) +
//
//		calculate(geo,f,((s - X) + Y) + Z)*geo.inv_volume(((s - X) + Y) + Z)
//
//		) * 0.125 * geo.volume(s);

		return calculate(geo,f, s) /** geo._impl::HodgeStarVolumeScale(s)*/;
	}

//	template<typename TM, typename TL, typename TR> void calculate(
//			_impl::InteriorProduct, nTuple<TR, G::ndims> const & v,
//			_Field<Domain<TM, VERTEX>, TL> const & f,
//			id_type   s)  = delete;

	template<typename TL,typename TR >
	static inline typename field_traits<_Field< _impl::InteriorProduct<EDGE, VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::InteriorProduct<EDGE, VERTEX, TL,TR>> const & expr,
			id_type s)
	{
		auto const & f =expr.lhs;
		auto const & v =expr.rhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(calculate(geo,f, s + X) - calculate(geo,f, s - X)) * 0.5 * v[0] //
		+ (calculate(geo,f, s + Y) - calculate(geo,f, s - Y)) * 0.5 * v[1]//
		+ (calculate(geo,f, s + Z) - calculate(geo,f, s - Z)) * 0.5 * v[2];
	}

	template<typename TL,typename TR >
	static inline typename field_traits<_Field< _impl::InteriorProduct<FACE, VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::InteriorProduct<FACE, VERTEX, TL,TR>> const & expr,
			id_type s)
	{
		auto const & f =expr.lhs;
		auto const & v =expr.rhs;

		size_t n = topology_type::component_number(s);

		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);
		return

		(calculate(geo,f, s + Y) + calculate(geo,f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(calculate(geo,f, s + Z) + calculate(geo,f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL,typename TR >
	static inline typename field_traits<_Field< _impl::InteriorProduct<VOLUME, VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::InteriorProduct<VOLUME, VERTEX, TL,TR>> const & expr,
			id_type s)
	{
		auto const & f =expr.lhs;
		auto const & v =expr.rhs;
		size_t n = topology_type::component_number(topology_type::dual(s));
		size_t D = topology_type::delta_index(topology_type::dual(s));

		return (calculate(geo,f, s + D) - calculate(geo,f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation

	template< size_t IL, typename T > static inline typename field_traits<T>::value_type
	calculate(geometry_type const & geo,_Field<_impl::MapTo<IL, IL, T>> const & f, id_type s)
	{
		return calculate(geo,f,s);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo<EDGE, VERTEX, T>>>::value_type
	map_to(geometry_type const & geo ,_Field<_impl::MapTo<EDGE, VERTEX, T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(calculate(geo,f,s))>::type, 3>(
				{

					(calculate(geo,f, s - X) + calculate(geo,f, s + X)) * 0.5, //
					(calculate(geo,f, s - Y) + calculate(geo,f, s + Y)) * 0.5,//
					(calculate(geo,f, s - Z) + calculate(geo,f, s + Z)) * 0.5

				});
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo< VERTEX, EDGE,T>>>::value_type
	map_to(geometry_type const & geo ,_Field<_impl::MapTo< VERTEX,EDGE, T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;
		auto n = topology_type::component_number(s);
		auto D = topology_type::delta_index(s);

		return ((calculate(geo,f, s - D)[n] + calculate(geo,f, s + D)[n]) * 0.5);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo< FACE, VERTEX,T>>>::value_type
	map_to(geometry_type const & geo ,_Field<_impl::MapTo< FACE,VERTEX, T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(calculate(geo,f,s))>::type, 3>(
				{	(

							calculate(geo,f, (s - Y) - Z) +

							calculate(geo,f, (s - Y) + Z) +

							calculate(geo,f, (s + Y) - Z) +

							calculate(geo,f, (s + Y) + Z)

					) * 0.25,

					(

							calculate(geo,f, (s - Z) - X) +

							calculate(geo,f, (s - Z) + X) +

							calculate(geo,f, (s + Z) - X) +

							calculate(geo,f, (s + Z) + X)

					) * 0.25,

					(

							calculate(geo,f, (s - X) - Y) +

							calculate(geo,f, (s - X) + Y) +

							calculate(geo,f, (s + X) - Y) +

							calculate(geo,f, (s + X) + Y)

					) * 0.25

				});
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo< VERTEX,FACE, T>>>::value_type
	map_to(geometry_type const & geo,_Field<_impl::MapTo< VERTEX, FACE, T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;

		auto n = topology_type::component_number(topology_type::dual(s));
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		return (

				(

						calculate(geo,f, (s - Y) - Z)[n] +

						calculate(geo,f, (s - Y) + Z)[n] +

						calculate(geo,f, (s + Y) - Z)[n] +

						calculate(geo,f, (s + Y) + Z)[n]

				) * 0.25

		);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo< FACE,VOLUME, T>>>::value_type
	map_to(geometry_type const & geo,_Field<_impl::MapTo< FACE,VOLUME,T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(calculate(geo,f,s))>::type,
		3>(
				{

					(calculate(geo,f, s - X) + calculate(geo,f, s + X)) * 0.5, //
					(calculate(geo,f, s - Y) + calculate(geo,f, s + Y)) * 0.5,//
					(calculate(geo,f, s - Z) + calculate(geo,f, s + Z)) * 0.5

				});
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo<VOLUME, FACE, T>>>::value_type
	map_to(geometry_type const & geo,_Field<_impl::MapTo< VOLUME,FACE,T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;

		auto n = topology_type::component_number(topology_type::dual(s));
		auto D = topology_type::delta_index(topology_type::dual(s));

		return ((calculate(geo,f, s - D)[n] + calculate(geo,f, s + D)[n]) * 0.5);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo<EDGE, VOLUME, T>>>::value_type
	map_to(geometry_type const & geo,_Field<_impl::MapTo<EDGE, VOLUME,T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return nTuple<
		typename std::remove_reference<decltype(calculate(geo,f,s))>::type,
		3>(
				{	(

							calculate(geo,f, (s - Y) - Z) +

							calculate(geo,f, (s - Y) + Z) +

							calculate(geo,f, (s + Y) - Z) +

							calculate(geo,f, (s + Y) + Z)

					) * 0.25,

					(

							calculate(geo,f, (s - Z) - X) +

							calculate(geo,f, (s - Z) + X) +

							calculate(geo,f, (s + Z) - X) +

							calculate(geo,f, (s + Z) + X)

					) * 0.25,

					(

							calculate(geo,f, (s - X) - Y) +

							calculate(geo,f, (s - X) + Y) +

							calculate(geo,f, (s + X) - Y) +

							calculate(geo,f, (s + X) + Y)

					) * 0.25,

				});
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo< VOLUME,EDGE, T>>>::value_type
	map_to(geometry_type const & geo,_Field<_impl::MapTo< VOLUME,EDGE,T>> const & expr,id_type s)
	{
		auto const & f= expr.lhs;
		auto n = topology_type::component_number(topology_type::dual(s));
		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);
		return (

				(

						calculate(geo,f, (s - Y) - Z)[n] +

						calculate(geo,f, (s - Y) + Z)[n] +

						calculate(geo,f, (s + Y) - Z)[n] +

						calculate(geo,f, (s + Y) + Z)[n]

				) * 0.25

		);
	}

	// For curl_pdx

	template<size_t N , typename T> static inline
	typename field_traits<_Field< _impl::PartialExteriorDerivative< N,EDGE , T > > >::value_type
	calculate(geometry_type const & geo,_Field< _impl::PartialExteriorDerivative< N,EDGE , T >> const & expr,
			id_type s)
	{
		auto const & f =expr.lhs;

		auto X = topology_type::delta_index(topology_type::dual(s));
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		Y = (topology_type::component_number(Y) == N) ? Y : 0UL;
		Z = (topology_type::component_number(Z) == N) ? Z : 0UL;

		return (calculate(geo,f, s + Y) - calculate(geo,f, s - Y))
		- (calculate(geo,f, s + Z) - calculate(geo,f, s - Z));
	}

	template<size_t N , typename T> static inline
	typename field_traits<_Field< _impl::PartialCodifferentialDerivative< N,FACE , T >> >::value_type
	calculate(geometry_type const & geo,_Field< _impl::PartialCodifferentialDerivative< N,FACE , T >> const & expr,
			id_type s)
	{
		auto const & f =expr.lhs;

		auto X = topology_type::delta_index(s);
		auto Y = topology_type::roate(X);
		auto Z = topology_type::inverse_roate(X);

		Y = (topology_type::component_number(Y) == N) ? Y : 0UL;
		Z = (topology_type::component_number(Z) == N) ? Z : 0UL;

		return (

				calculate(geo,f, s + Y) * (geo.dual_volume(s + Y))      //
				- calculate(geo,f, s - Y) * (geo.dual_volume(s - Y))//
				- calculate(geo,f, s + Z) * (geo.dual_volume(s + Z))//
				+ calculate(geo,f, s - Z) * (geo.dual_volume(s - Z))//

		) * geo.inv_dual_volume(s);
	}

};

}
// namespace simpla

#endif /* FDM_H_ */
