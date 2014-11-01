/*
 * calculus.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_
#include <type_traits>
#include "../utilities/primitives.h"
#include "../utilities/ntuple.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/sp_functional.h"
#include "../utilities/constant_ops.h"
#include "../utilities/expression_template.h"
#include "domain.h"

namespace simpla
{
template<typename ... > class _Field;
template<typename, size_t> class Domain;
template<typename ... > class Expression;

/// \defgroup  ExteriorAlgebra Exterior algebra
/// @{
namespace _impl
{
struct HodgeStar
{
};

struct InteriorProduct
{
};

struct Wedge
{
};

struct ExteriorDerivative
{
};

struct CodifferentialDerivative
{
};

struct MapTo
{
};
} //namespace _impl
template<typename ...> struct field_traits;

//template<typename TManifold, size_t IFORM, typename ...Others>
//struct field_traits<_Field<Domain<TManifold, IFORM>, Others...>>
//{
//	static constexpr size_t ndims = TManifold::ndims;
//
//	static constexpr size_t iform = IFORM;
//};
//
//template<typename TOP, typename TL>
//struct field_traits<_Field<Expression<TOP, TL> > >
//{
//	static constexpr size_t ndims = field_traits<TL>::ndims;
//	static constexpr size_t iform = field_traits<TL>::iform;
//};
//
//template<typename TOP, typename TL, typename TR>
//struct field_traits<_Field<Expression<TOP, TL, TR> > >
//{
//	static constexpr size_t ndims = field_traits<TL>::ndims;
//	static constexpr size_t iform = field_traits<TL>::iform;
//};
template<typename TL>
struct field_traits<_Field<Expression<_impl::HodgeStar, TL> > >
{
	static constexpr size_t ndims = field_traits<TL>::ndims;
	static constexpr size_t iform = ndims - field_traits<TL>::iform;
	typedef typename field_traits<TL>::manifold_type manifold_type;
	typedef Domain<manifold_type, iform> domain_type;

	typedef _Field<Expression<_impl::HodgeStar, TL> > field_type;

//	static domain_type get_domain(field_type const &f)
//	{
//		return (
//				make_domain<iform>(
//						field_traits<TL>::get_domain(f.lhs).manifold()));
//	}
//
//	static domain_type get_domain(TL const &f)
//	{
//		return (
//				make_domain<iform>(field_traits<TL>::get_domain(f).manifold()));
//	}

};

template<typename TL, typename TR>
struct field_traits<_Field<Expression<_impl::InteriorProduct, TL, TR> > >
{
	static constexpr size_t ndims = field_traits<TL>::ndims;

	static constexpr size_t iform = field_traits<TL>::iform - 1;

	typedef typename field_traits<TL>::manifold_type manifold_type;

	typedef Domain<manifold_type, iform> domain_type;

	typedef _Field<Expression<_impl::InteriorProduct, TL, TR> > field_type;

//	static domain_type get_domain(field_type const &f)
//	{
//		return (
//				make_domain<iform>(
//						field_traits<TL>::get_domain(f.lhs).manifold()));
//	}
//
//	static domain_type get_domain(TL const &f, TR const &)
//	{
//		return (
//				make_domain<iform>(field_traits<TL>::get_domain(f).manifold()));
//	}

};

template<typename TL, typename TR>
struct field_traits<_Field<Expression<_impl::Wedge, TL, TR> > >
{
	static constexpr size_t ndims = field_traits<TL>::ndims;

	static constexpr size_t iform = field_traits<TL>::iform
			+ field_traits<TR>::iform;

	typedef typename field_traits<TL>::manifold_type manifold_type;

	typedef Domain<manifold_type, iform> domain_type;

	typedef _Field<Expression<_impl::Wedge, TL, TR> > field_type;

//	static domain_type get_domain(field_type const &f)
//	{
//		return (
//				make_domain<iform>(
//						field_traits<TL>::get_domain(f.lhs).manifold()));
//	}
//
//	static domain_type get_domain(TL const & l, TR const & r)
//	{
//		return (
//				make_domain<iform>(field_traits<TL>::get_domain(l).manifold()));
//	}

};

template<typename TL>
struct field_traits<_Field<Expression<_impl::ExteriorDerivative, TL> > >
{
	static constexpr size_t ndims = field_traits<TL>::ndims;

	static constexpr size_t iform = field_traits<TL>::iform + 1;

	typedef typename field_traits<TL>::manifold_type manifold_type;

	typedef Domain<manifold_type, iform> domain_type;

	typedef _Field<Expression<_impl::ExteriorDerivative, TL> > field_type;

//	static domain_type get_domain(field_type const &f)
//	{
//		return (
//				make_domain<iform>(
//						field_traits<TL>::get_domain(f.lhs).manifold()));
//	}
//
//	static domain_type get_domain(TL const &f)
//	{
//		return (
//				make_domain<iform>(field_traits<TL>::get_domain(f).manifold()));
//	}

};

template<typename TL>
struct field_traits<_Field<Expression<_impl::CodifferentialDerivative, TL> > >
{

	static constexpr size_t ndims = field_traits<TL>::ndims;

	static constexpr size_t iform = field_traits<TL>::iform - 1;

	typedef typename field_traits<TL>::manifold_type manifold_type;

	typedef Domain<manifold_type, iform> domain_type;

	typedef _Field<Expression<_impl::CodifferentialDerivative, TL> > field_type;

//	static domain_type get_domain(field_type const &f)
//	{
//		return (
//				make_domain<iform>(
//						field_traits<TL>::get_domain(f.lhs).manifold()));
//	}
//
//	static domain_type get_domain(TL const &f)
//	{
//		return (
//				make_domain<iform>(field_traits<TL>::get_domain(f).manifold()));
//	}
};

template<typename ... T>
inline _Field<Expression<_impl::HodgeStar, _Field<T...>>> hodge_star(_Field<T...> const & f)
{
	return ((_Field<Expression<_impl::HodgeStar, _Field<T...>>>(f)));
}

template<typename ... T, typename TR>
inline _Field<Expression<_impl::Wedge, _Field<T...>, TR>> wedge(
		_Field<T...> const & l, TR const & r)
{
	return (_Field<Expression<_impl::Wedge, _Field<T...>, TR>>(l, r));
}

template<typename ...T, typename TR>
inline auto interior_product(TR const & v, _Field<T...> const & l)
DECL_RET_TYPE ((_Field<Expression<_impl::InteriorProduct,
				TR,_Field<T...> > >(v,l)))

template<typename ...T>
inline auto exterior_derivative(_Field<T...> const & f)
DECL_RET_TYPE((_Field<Expression<_impl::ExteriorDerivative,
				_Field<T...> > >(f )))

template<typename ...T>
inline auto codifferential_derivative(_Field<T...> const & f)
DECL_RET_TYPE((_Field<Expression<_impl::CodifferentialDerivative,
				_Field<T...> > >(f )))

template<typename ...T>
inline auto operator*(_Field<T...> const & f)
DECL_RET_TYPE((hodge_star(f)))
;
template<typename ... T>
inline auto d(_Field<T...> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )
;

template<typename ... T>
inline auto delta(_Field<T...> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
;

template<size_t ndims, typename TL, typename ...T>
inline auto iv(nTuple<TL, ndims> const & v, _Field<T...> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )
;

template<typename ...T1, typename ... T2>
inline auto operator^(_Field<T1...> const & lhs, _Field<T2...> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )
;

///  @}

///  \defgroup  VectorAlgebra Vector algebra
///  @{
template<typename ...TL, typename TR> inline auto inner_product(
		_Field<TL...> const & lhs, TR const & rhs)
		DECL_RET_TYPE(wedge (lhs,hodge_star( rhs) ))
;

template<typename ...TL, typename TR> inline auto dot(_Field<TL...> const & lhs,
		TR const & rhs)
		DECL_RET_TYPE(wedge(lhs , hodge_star(rhs) ))
;
template<typename ...TL, typename ...TR> inline auto cross(
		_Field<TL...> const & lhs, _Field<TR...> const & rhs)
		ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<TL...>>::iform==EDGE),
				wedge(lhs , rhs ))
;

template<typename ...TL, typename ...TR> inline auto cross(
		_Field<TL...> const & lhs, _Field<TR...> const & rhs)
		ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<TL...>>::iform==FACE),
				hodge_star(wedge(hodge_star(lhs) , hodge_star(rhs) )))

template<typename TL, typename ... TR> inline auto dot(nTuple<TL, 3> const & v,
		_Field<TR...> const & f)
		DECL_RET_TYPE( (interior_product(v, f)))

template<typename ...TL, typename TR> inline auto dot(_Field<TL...> const & f,
		nTuple<TR, 3> const & v)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename ... TL, typename TR> inline auto cross(
		_Field<TL...> const & f, nTuple<TR, 3> const & v)
		DECL_RET_TYPE( (interior_product(v, hodge_star(f))))
;

template<typename ... T, typename TL> inline auto cross(_Field<T...> const & f,
		nTuple<TL, 3> const & v)
		DECL_RET_TYPE((interior_product(v, f)))
;

//template<typename ...T>
//inline auto grad(_Field<T...> const & f)
//DECL_RET_TYPE( ( exterior_derivative(f)))
//;

template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==VERTEX),
		(exterior_derivative(f)))

template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==VOLUME),
		((codifferential_derivative(-f))) )

template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		(exterior_derivative(f)))
;

template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==EDGE),
		(codifferential_derivative(-f)))

template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==EDGE),
		(exterior_derivative(f)))

template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		((codifferential_derivative(-f))) )

///   @}

/////  \ingroup  FETL
/////  \defgroup  NonstandardOperations Non-standard operations
/////   @{
//template<typename TM, typename TR> inline auto CurlPDX(
//		_Field<Domain<TM, EDGE>, TR> const & f)
//				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<size_t ,0>())))
//;
//
//template<typename TM, typename TR> inline auto CurlPDY(
//		_Field<Domain<TM, EDGE>, TR> const & f)
//				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<size_t ,1>())))
//;
//
//template<typename TM, typename TR> inline auto CurlPDZ(
//		_Field<Domain<TM, EDGE>, TR> const & f)
//				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<size_t ,2>())))
//;
//
//template<typename TM, typename TR> inline auto CurlPDX(
//		_Field<Domain<TM, FACE>, TR> const & f)
//				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<size_t ,0>())))
//;
//
//template<typename ...T> inline auto CurlPDY(
//		_Field<T...> const & f)
//				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<size_t ,1>())))
//;
//
//template<typename TM, typename TR>
//inline auto CurlPDZ(
//		_Field<Domain<TM, FACE>, TR> const & f)
//				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<size_t ,2>())))
//;

//template<size_t IL, typename TM, size_t IR, typename TR>
//inline auto MapTo(
//		_Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (_Field<Domain<TM,IL>, _Field<MAPTO,std::integral_constant<size_t ,IL>,_Field<Domain<TM, IR>, TR> > >(std::integral_constant<size_t ,IL>(), f)))
//;
//

template<size_t IL, typename ...T>
inline _Field<
		Expression<_impl::MapTo, std::integral_constant<size_t, IL>,
				_Field<T...>>> map_to(_Field<T...> const & f)
{
	return ((_Field<
			Expression<_impl::MapTo, std::integral_constant<size_t, IL>,
					_Field<T...>>>(std::integral_constant<size_t, IL>(), f)));
}

///   @}

}
// namespace simpla

#endif /* CALCULUS_H_ */
