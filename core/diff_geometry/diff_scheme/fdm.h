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
#include "../diff_geometry_common.h"

namespace simpla
{

template<typename ... > class _Field;
template<typename, size_t> class Domain;

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
	typedef typename geometry_type::index_type index_type;
	typedef Real scalar_type;
	static constexpr size_t NUM_OF_COMPONENT_TYPE = G::ndims + 1;
	static constexpr size_t ndims = G::ndims;

	FiniteDiffMethod()
	{
	}

	FiniteDiffMethod(this_type const & r) = default;

protected:
	~FiniteDiffMethod() = default;
public:
	this_type & operator=(this_type const &) = default;

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
	Real calculate(geometry_type const & geo, Real v, Others &&... s) const
	{
		return v;
	}

	template<typename ...Others>
	int calculate(geometry_type const & geo, int v, Others &&... s) const
	{
		return v;
	}

	template<typename ...Others>
	std::complex<Real> calculate(geometry_type const & geo,
			std::complex<Real> v, Others &&... s) const
	{
		return v;
	}

	template<typename T, size_t ...N, typename ...Others>
	nTuple<T, N...> const& calculate(geometry_type const & geo,
			nTuple<T, N...> const& v, Others &&... s) const
	{
		return v;
	}

	template<typename ...T, typename ...Others>
	inline typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type
	calculate(geometry_type const & geo,nTuple<Expression<T...>> const & v, Others &&... s) const
	{
		typename nTuple_traits<nTuple<Expression<T...>>> ::primary_type res;
		res=v;
		return std::move(res);
	}

	template<typename TC, typename TD, typename ... Others>
	inline typename field_traits<_Field<TC, TD> >::value_type
	calculate(geometry_type const & geo,_Field<TC, TD> const &f, Others && ... s) const
	{
		return f.get(std::forward<Others>(s)...);
	}

	template<typename TOP, typename TL, typename TR, typename ...Others>
	inline typename field_traits< _Field<Expression<TOP, TL, TR>>>::value_type
	calculate(geometry_type const & geo,_Field<Expression<TOP, TL, TR>> const &f, Others &&... s) const
	{
		return f.op_(calculate( geo,f.lhs,std::forward<Others>(s)...),
				calculate(geo,f.rhs,std::forward<Others>(s)...));
	}

	template<typename TOP, typename TL, typename ...Others>
	inline typename field_traits< _Field<Expression<TOP, TL,std::nullptr_t>>>::value_type
	calculate(geometry_type const & geo,_Field<Expression<TOP, TL,std::nullptr_t>> const &f, Others &&... s) const
	{
		return f.op_(calculate(geo,f.lhs,std::forward<Others>(s)...) );
	}

	template<typename T,typename TI>
	inline typename field_traits<_Field<_impl::ExteriorDerivative< VERTEX,T> >>::value_type
	calculate(geometry_type const & geo,_Field<_impl::ExteriorDerivative<VERTEX,T> > const & f, TI s) const
	{
		auto D = geo.delta_index(s);

		return (calculate(geo,f.lhs, s + D) * geo.volume(s + D)
				- calculate(geo,f.lhs, s - D) * geo.volume(s - D)) * geo.inv_volume(s);
	}

	template<typename T,typename TI>
	inline typename field_traits<_Field<_impl::ExteriorDerivative< EDGE,T> >>::value_type
	calculate(geometry_type const & geo,_Field<_impl::ExteriorDerivative<EDGE,T> > const & expr, TI s) const
	{
		auto const & f=expr.lhs;

		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return (

				(

						calculate(geo,f, s + Y) * geo.volume(s + Y) //
						- calculate(geo,f, s - Y) * geo.volume(s - Y)//

				) - (

						calculate(geo,f, s + Z) * geo.volume(s + Z)//
						- calculate(geo,f, s - Z) * geo.volume(s - Z)//

				)

		) * geo.inv_volume(s);

	}

	template<typename T,typename TI>
	inline typename field_traits<_Field<_impl::ExteriorDerivative< FACE,T> >>::value_type
	calculate(geometry_type const & geo,_Field<_impl::ExteriorDerivative<FACE,T> > const & expr, TI s) const
	{
		auto const & f=expr.lhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return (

				calculate(geo,f, s + X) * geo.volume(s + X)

				- calculate(geo,f, s - X) * geo.volume(s - X) //
				+ calculate(geo,f, s + Y) * geo.volume(s + Y)//
				- calculate(geo,f, s - Y) * geo.volume(s - Y)//
				+ calculate(geo,f, s + Z) * geo.volume(s + Z)//
				- calculate(geo,f, s - Z) * geo.volume(s - Z)//

		) * geo.inv_volume(s)

		;
	}
//
////	template<typename TM, size_t IL, typename TL> void calculate(
////			_impl::ExteriorDerivative, _Field<Domain<TM, IL>, TL> const & f,
////			index_type s) const = delete;
////
////	template<typename TM, size_t IL, typename TL> void calculate(
////			_impl::CodifferentialDerivative,
////			_Field<TL...> const & f, index_type s) const = delete;

	template<typename T >
	inline typename field_traits<_Field< _impl::CodifferentialDerivative< EDGE, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::CodifferentialDerivative< EDGE, T>> const & expr,
			index_type s) const
	{
		auto const & f=expr.lhs;

		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		-(

				calculate(geo,f, s + X) * geo.dual_volume(s + X)

				- calculate(geo,f, s - X) * geo.dual_volume(s - X)

				+ calculate(geo,f, s + Y) * geo.dual_volume(s + Y)

				- calculate(geo,f, s - Y) * geo.dual_volume(s - Y)

				+ calculate(geo,f, s + Z) * geo.dual_volume(s + Z)

				- calculate(geo,f, s - Z) * geo.dual_volume(s - Z)

		) * geo.inv_dual_volume(s)

		;

	}

	template<typename T >
	inline typename field_traits<_Field< _impl::CodifferentialDerivative< FACE, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::CodifferentialDerivative< FACE, T>> const & expr,
			index_type s) const
	{
		auto const & f=expr.lhs;
		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return

		-(
				(calculate(geo,f, s + Y) * (geo.dual_volume(s + Y))
						- calculate(geo,f, s - Y) * (geo.dual_volume(s - Y)))

				- (calculate(geo,f, s + Z) * (geo.dual_volume(s + Z))
						- calculate(geo,f, s - Z) * (geo.dual_volume(s - Z)))

		) * geo.inv_dual_volume(s)

		;
	}

	template<typename T >
	inline typename field_traits<_Field< _impl::CodifferentialDerivative< VOLUME, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::CodifferentialDerivative< VOLUME, T>> const & expr,
			index_type s) const
	{
		auto const & f=expr.lhs;
		auto D = geo.delta_index(geo.dual(s));
		return

		-(

				calculate(geo,f, s + D) * (geo.dual_volume(s + D)) //
				- calculate(geo,f, s - D) * (geo.dual_volume(s - D))

		) * geo.inv_dual_volume(s)

		;
	}

////***************************************************************************************************
//
////! Form<IR> ^ Form<IR> => Form<IR+IL>

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,VERTEX,TL,TR>> const & expr,
			index_type s) const
	{
		return (calculate(expr.lhs, s) * calculate(expr.rhs, s));
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,EDGE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,EDGE,TL,TR>> const & expr,
			index_type s) const
	{
		auto X = geo.delta_index(s);

		return (calculate(expr.lhs, s - X) + calculate(expr.lhs, s + X)) * 0.5
		* calculate(expr.rhs, s);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,FACE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,FACE,TL,TR>> const & expr,
			index_type s) const
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		return (

				calculate(geo,l, (s - Y) - Z) +

				calculate(geo,l, (s - Y) + Z) +

				calculate(geo,l, (s + Y) - Z) +

				calculate(geo,l, (s + Y) + Z)

		) * 0.25 * calculate(geo,r, s);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VERTEX,VOLUME,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VERTEX,VOLUME,TL,TR>> const & expr,
			index_type s) const
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
	inline typename field_traits<_Field<_impl::Wedge<EDGE,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<EDGE,VERTEX,TL,TR>> const & expr,
			index_type s) const
	{

		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto X = geo.delta_index(s);
		return calculate(geo,l, s) * (calculate(geo,r, s - X) + calculate(geo,r, s + X))
		* 0.5;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<EDGE,EDGE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<EDGE,EDGE,TL,TR>> const & expr,
			index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;

		auto Y = geo.delta_index(geo.roate(geo.dual(s)));
		auto Z = geo.delta_index(geo.inverse_roate(geo.dual(s)));

		return ((calculate(geo,l, s - Y) + calculate(geo,l, s + Y))
				* (calculate(geo,l, s - Z) + calculate(geo,l, s + Z)) * 0.25);
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<EDGE,FACE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<EDGE,FACE,TL,TR>> const & expr,
			index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

				(calculate(geo,l, (s - Y) - Z) + calculate(geo,l, (s - Y) + Z)
						+ calculate(geo,l, (s + Y) - Z) + calculate(geo,l, (s + Y) + Z))
				* (calculate(geo,r, s - X) + calculate(geo,r, s + X))
				+

				(calculate(geo,l, (s - Z) - X) + calculate(geo,l, (s - Z) + X)
						+ calculate(geo,l, (s + Z) - X)
						+ calculate(geo,l, (s + Z) + X))
				* (calculate(geo,r, s - Y) + calculate(geo,r, s + Y))
				+

				(calculate(geo,l, (s - X) - Y) + calculate(geo,l, (s - X) + Y)
						+ calculate(geo,l, (s + X) - Y)
						+ calculate(geo,l, (s + X) + Y))
				* (calculate(geo,r, s - Z) + calculate(geo,r, s + Z))

		) * 0.125;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<FACE,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<FACE,VERTEX,TL,TR>> const & expr,
			index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto Y = geo.delta_index(geo.roate(geo.dual(s)));
		auto Z = geo.delta_index(geo.inverse_roate(geo.dual(s)));

		return calculate(geo,l, s)
		* (calculate(geo,r, (s - Y) - Z) + calculate(geo,r, (s - Y) + Z)
				+ calculate(geo,r, (s + Y) - Z)
				+ calculate(geo,r, (s + Y) + Z)) * 0.25;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<FACE,EDGE,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<FACE,EDGE,TL,TR>> const & expr,
			index_type s) const
	{
		auto const & l =expr.lhs;
		auto const & r =expr.rhs;
		auto X = geo.DI(0, s);
		auto Y = geo.DI(1, s);
		auto Z = geo.DI(2, s);

		return

		(

				(calculate(geo,r, (s - Y) - Z) + calculate(geo,r, (s - Y) + Z)
						+ calculate(geo,r, (s + Y) - Z) + calculate(geo,r, (s + Y) + Z))
				* (calculate(geo,l, s - X) + calculate(geo,l, s + X))

				+ (calculate(geo,r, (s - Z) - X) + calculate(geo,r, (s - Z) + X)
						+ calculate(geo,r, (s + Z) - X)
						+ calculate(geo,r, (s + Z) + X))
				* (calculate(geo,l, s - Y) + calculate(geo,l, s + Y))

				+ (calculate(geo,r, (s - X) - Y) + calculate(geo,r, (s - X) + Y)
						+ calculate(geo,r, (s + X) - Y)
						+ calculate(geo,r, (s + X) + Y))
				* (calculate(geo,l, s - Z) + calculate(geo,l, s + Z))

		) * 0.125;
	}

	template<typename TL,typename TR>
	inline typename field_traits<_Field<_impl::Wedge<VOLUME,VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::Wedge<VOLUME,VERTEX,TL,TR>> const & expr,
			index_type s) const
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
	inline typename field_traits<_Field< _impl::HodgeStar< VOLUME, T> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::HodgeStar< VOLUME, T>> const & expr,
			index_type s) const
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
//			index_type s) const = delete;

	template<typename TL,typename TR >
	inline typename field_traits<_Field< _impl::InteriorProduct<EDGE, VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::InteriorProduct<EDGE, VERTEX, TL,TR>> const & expr,
			index_type s) const
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
	inline typename field_traits<_Field< _impl::InteriorProduct<FACE, VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::InteriorProduct<FACE, VERTEX, TL,TR>> const & expr,
			index_type s) const
	{
		auto const & f =expr.lhs;
		auto const & v =expr.rhs;

		size_t n = geo.component_number(s);

		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);
		return

		(calculate(geo,f, s + Y) + calculate(geo,f, s - Y)) * 0.5 * v[(n + 2) % 3] -

		(calculate(geo,f, s + Z) + calculate(geo,f, s - Z)) * 0.5 * v[(n + 1) % 3];
	}

	template<typename TL,typename TR >
	inline typename field_traits<_Field< _impl::InteriorProduct<VOLUME, VERTEX,TL,TR> > >::value_type
	calculate(geometry_type const & geo,_Field<_impl::InteriorProduct<VOLUME, VERTEX, TL,TR>> const & expr,
			index_type s) const
	{
		auto const & f =expr.lhs;
		auto const & v =expr.rhs;
		size_t n = geo.component_number(geo.dual(s));
		size_t D = geo.delta_index(geo.dual(s));

		return (calculate(geo,f, s + D) - calculate(geo,f, s - D)) * 0.5 * v[n];
	}

//**************************************************************************************************
// Non-standard operation

	template< size_t IL, typename T > inline typename field_traits<T>::value_type
	calculate(geometry_type const & geo,_Field<_impl::MapTo<IL, IL, T>> const & f, index_type s) const
	{
		return calculate(geo,f,s);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo<EDGE, VERTEX, T>>>::value_type
	map_to(geometry_type const & geo ,_Field<_impl::MapTo<EDGE, VERTEX, T>> const & expr,index_type s)
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
	map_to(geometry_type const & geo ,_Field<_impl::MapTo< VERTEX,EDGE, T>> const & expr,index_type s)
	{
		auto const & f= expr.lhs;
		auto n = geo.component_number(s);
		auto D = geo.delta_index(s);

		return ((calculate(geo,f, s - D)[n] + calculate(geo,f, s + D)[n]) * 0.5);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo< FACE, VERTEX,T>>>::value_type
	map_to(geometry_type const & geo ,_Field<_impl::MapTo< FACE,VERTEX, T>> const & expr,index_type s)
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
	map_to(geometry_type const & geo,_Field<_impl::MapTo< VERTEX, FACE, T>> const & expr,index_type s)
	{
		auto const & f= expr.lhs;

		auto n = geo.component_number(geo.dual(s));
		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

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
	map_to(geometry_type const & geo,_Field<_impl::MapTo< FACE,VOLUME,T>> const & expr,index_type s)
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
	map_to(geometry_type const & geo,_Field<_impl::MapTo< VOLUME,FACE,T>> const & expr,index_type s)
	{
		auto const & f= expr.lhs;

		auto n = geo.component_number(geo.dual(s));
		auto D = geo.delta_index(geo.dual(s));

		return ((calculate(geo,f, s - D)[n] + calculate(geo,f, s + D)[n]) * 0.5);
	}

	template< typename T>
	typename field_traits<_Field<_impl::MapTo<EDGE, VOLUME, T>>>::value_type
	map_to(geometry_type const & geo,_Field<_impl::MapTo<EDGE, VOLUME,T>> const & expr,index_type s)
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
	map_to(geometry_type const & geo,_Field<_impl::MapTo< VOLUME,EDGE,T>> const & expr,index_type s)
	{
		auto const & f= expr.lhs;
		auto n = geo.component_number(geo.dual(s));
		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);
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

	template<size_t N , typename T> inline
	typename field_traits<_Field< _impl::PartialExteriorDerivative< N,EDGE , T > > >::value_type
	calculate(geometry_type const & geo,_Field< _impl::PartialExteriorDerivative< N,EDGE , T >> const & expr,
			index_type s)
	{
		auto const & f =expr.lhs;

		auto X = geo.delta_index(geo.dual(s));
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		Y = (geo.component_number(Y) == N) ? Y : 0UL;
		Z = (geo.component_number(Z) == N) ? Z : 0UL;

		return (calculate(geo,f, s + Y) - calculate(geo,f, s - Y))
		- (calculate(geo,f, s + Z) - calculate(geo,f, s - Z));
	}

	template<size_t N , typename T> inline
	typename field_traits<_Field< _impl::PartialCodifferentialDerivative< N,FACE , T >> >::value_type
	calculate(geometry_type const & geo,_Field< _impl::PartialCodifferentialDerivative< N,FACE , T >> const & expr,
			index_type s)
	{
		auto const & f =expr.lhs;

		auto X = geo.delta_index(s);
		auto Y = geo.roate(X);
		auto Z = geo.inverse_roate(X);

		Y = (geo.component_number(Y) == N) ? Y : 0UL;
		Z = (geo.component_number(Z) == N) ? Z : 0UL;

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
