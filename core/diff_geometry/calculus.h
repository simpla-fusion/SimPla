/**
 * @file calculus.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_
#include <type_traits>

#include "../utilities/utilities.h"
#include "../utilities/sp_functional.h"
#include "../design_pattern/expression_template.h"

namespace simpla
{

template<typename ... > class _Field;
template<typename, typename, typename > class Expression;

/**
 * @ingroup diff_geo
 *  @{
 *   @addtogroup  linear_algebra Linear Algebra
 *    @brief  This module collects linear algebra operations between field/forms
 *
 *    \note  Linear algebra is the branch of mathematics concerning vector spaces and linear mappings between such spaces. --wiki
 *    \note \f$\Omega^n\f$ means fields/forms on N-dimensional manifold \f$M\f$.
 *  @}
 *
 */
/**
 *  @ingroup linear_algebra
 *  @addtogroup exterior_algebra  Exterior algebra of forms
 *  @{
 *
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{N-n}\f$ =HodgeStar(\f$\Omega^n\f$ )	| hodge star, abbr. operator *
 *  \f$\Omega^{m+n}\f$ =Wedge(\f$\Omega^m\f$ ,\f$\Omega^m\f$  )	| wedge product, abbr. operator^
 *  \f$\Omega^{n-1}\f$ =InteriorProduct(Vector Field ,\f$\Omega^n\f$  )	| interior product, abbr. iv
 *  \f$\Omega^{N}\f$ =InnerProduct(\f$\Omega^m\f$ ,\f$\Omega^m\f$ ) | inner product,
 *
 */
namespace _impl
{
template<size_t IL, typename T> struct HodgeStar
{
};
template<size_t IL, size_t IR, typename TL, typename TR> struct InteriorProduct
{
};
template<size_t IL, size_t IR, typename TL, typename TR> struct Wedge
{
};
} //namespace _impl

template<size_t IL, typename T>
struct _Field<_impl::HodgeStar<IL, T>> : public Expression<
		_impl::HodgeStar<IL, T>, T, std::nullptr_t>
{
	using Expression<_impl::HodgeStar<IL, T>, T, std::nullptr_t>::Expression;
};

template<size_t IL, size_t IR, typename TL, typename TR>
struct _Field<_impl::InteriorProduct<IL, IR, TL, TR>> : public Expression<
		_impl::InteriorProduct<IL, IR, TL, TR>, TL, TR>
{
	using Expression<_impl::InteriorProduct<IL, IR, TL, TR>, TL, TR>::Expression;
};

template<size_t IL, size_t IR, typename TL, typename TR>
struct _Field<_impl::Wedge<IL, IR, TL, TR>> : public Expression<
		_impl::Wedge<IL, IR, TL, TR>, TL, TR>
{
	using Expression<_impl::Wedge<IL, IR, TL, TR>, TL, TR>::Expression;
};

template<typename ...> struct field_traits;

template<size_t IL, typename T>
struct field_traits<_Field<_impl::HodgeStar<IL, T> > >
{
private:
	static constexpr size_t NDIMS = field_traits<T>::ndims;
//	static constexpr size_t IL = field_traits<T>::iform;
public:

	static const size_t ndims = NDIMS > -IL ? NDIMS : 0;
	static const size_t iform = NDIMS - IL;

	typedef typename field_traits<T>::value_type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

template<size_t IL, size_t IR, typename TL, typename TR>
struct field_traits<_Field<_impl::InteriorProduct<IL, IR, TL, TR> > >
{
private:
	static constexpr size_t NDIMS = sp_max<size_t, field_traits<TL>::ndims,
			field_traits<TL>::ndims>::value;
//	static constexpr size_t IL = field_traits<TL>::iform;
//	static constexpr size_t IR = field_traits<TR>::iform;
//
	typedef typename field_traits<TL>::value_type l_type;
	typedef typename field_traits<TR>::value_type r_type;

public:
	static const size_t ndims = sp_max<size_t, IL, IR>::value > 0 ? NDIMS : 0;
	static const size_t iform = sp_max<size_t, IL, IR>::value - 1;

	typedef typename sp_result_of<_impl::multiplies(l_type, r_type)>::type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

template<size_t IL, size_t IR, typename TL, typename TR>
struct field_traits<_Field<_impl::Wedge<IL, IR, TL, TR> > >
{
private:
	static constexpr size_t NDIMS = sp_max<size_t, field_traits<TL>::ndims,
			field_traits<TL>::ndims>::value;
//	static constexpr size_t IL = field_traits<TL>::iform;
//	static constexpr size_t IR = field_traits<TR>::iform;

	typedef typename field_traits<TL>::value_type l_type;
	typedef typename field_traits<TR>::value_type r_type;
public:
	static const size_t ndims = IL + IR <= NDIMS ? NDIMS : 0;
	static const size_t iform = IL + IR;

	typedef typename sp_result_of<_impl::multiplies(l_type, r_type)>::type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

template<typename T>
inline auto hodge_star(T const & f)
DECL_RET_TYPE(( _Field<_impl::HodgeStar<
				field_traits<T>::iform , T >>(f)))

template<typename TL, typename TR>
inline auto wedge(TL const & l, TR const & r)
DECL_RET_TYPE((_Field< _impl::Wedge<
				field_traits<TL>::iform, field_traits<TR>::iform
				, TL, TR> > (l, r)))

template<typename TL, typename TR>
inline auto interior_product(TL const & l, TR const & r)
DECL_RET_TYPE((_Field< _impl::InteriorProduct<
				field_traits<TL>::iform, field_traits<TR>::iform
				, TL, TR>> (l, r)))

template<typename ...T>
inline auto operator*(_Field<T...> const & f)
DECL_RET_TYPE((hodge_star(f)))

template<size_t ndims, typename TL, typename ...T>
inline auto iv(nTuple<TL, ndims> const & v, _Field<T...> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )

template<typename ...T1, typename ... T2>
inline auto operator^(_Field<T1...> const & lhs, _Field<T2...> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )
/** @} */
/**
 * @ingroup linear_algebra
 * @addtogroup  vector_algebra   Linear algebra of vector fields
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$  	| negate operation
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$  	| positive operation
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ +\f$\Omega^n\f$ 	| add
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ -\f$\Omega^n\f$ 	| subtract
 *  \f$\Omega^n\f$ =\f$\Omega^n\f$ *Scalar  	| multiply
 *  \f$\Omega^n\f$ = Scalar * \f$\Omega^n\f$  	| multiply
 *  \f$\Omega^n\f$ = \f$\Omega^n\f$ / Scalar  	| divide
 *
 */
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
;
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
/** @} */
/**
 * @ingroup linear_algebra
 * @addtogroup linear_map Linear map between forms/fields.
 * @{
 *
 *   Map between vector form  and scalar form
 *
 *  Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{1}\f$ =MapTo(\f${V\Omega}^0\f$ )	| map vector 0-form to 1-form
 *  \f${V\Omega}^{0}\f$ =MapTo(\f$\Omega^1\f$ )	| map 1-form to vector 0-form
 *
 *  \f{eqnarray*}{
 *  R &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega+\Omega_{s}\right)}\\
 *  L &=& 1+\sum_{s}\frac{\omega_{ps}^{2}}{\omega\left(\omega-\Omega_{s}\right)}\\
 *  P &=& 1-\sum_{s}\frac{\omega_{ps}^{2}}{\omega^{2}}
 *  \f}
 */
namespace _impl
{

template<size_t IL, size_t IR, typename T>
struct MapTo
{
};

}  // namespace _impl

template<size_t IL, size_t IR, typename T>
struct field_traits<_Field<_impl::MapTo<IL, IR, T> > >
{
	static constexpr size_t NDIMS = field_traits<T>::ndims;
public:
	static const size_t ndims = NDIMS;
	static const size_t iform = IR;

	typedef typename field_traits<T>::value_type value_type;
};
template<size_t IR, typename T>
inline _Field<_impl::MapTo<field_traits<T>::iform, IR, T>> map_to(T const & f)
{
	return std::move((_Field<_impl::MapTo<field_traits<T>::iform, IR, T>>(f)));
}

/** @} */
/**
 * @ingroup diff_geo
 * @{
 * @addtogroup  diff_calculus Differential calculus
 * @brief  This module collects differential calculus on fields or forms
 * @}
 */
/**
 * @ingroup diff_calculus
 * @addtogroup dif_calculus_form Differential calculus of forms
 * @{
 *   Pseudo-Signature  			| Semantics
 *  -------------------------------|--------------
 *  \f$\Omega^{n-1}\f$ =ExteriorDerivative(\f$\Omega^n\f$ )	| Exterior Derivative, abbr. d
 *  \f$\Omega^{n+1}\f$ =Codifferential(\f$\Omega^n\f$ )	| Codifferential Derivative, abbr. delta
 *
 */
namespace _impl
{

template<size_t IL, typename T> struct ExteriorDerivative
{
};
template<size_t IL, typename T> struct CodifferentialDerivative
{
};

}  // namespace _impl
template<size_t IL, typename T>
struct _Field<_impl::ExteriorDerivative<IL, T>> : public Expression<
		_impl::ExteriorDerivative<IL, T>, T, std::nullptr_t>
{
	using Expression<_impl::ExteriorDerivative<IL, T>, T, std::nullptr_t>::Expression;
};

template<size_t IL, typename T>
struct _Field<_impl::CodifferentialDerivative<IL, T>> : public Expression<
		_impl::CodifferentialDerivative<IL, T>, T, std::nullptr_t>
{
	using Expression<_impl::CodifferentialDerivative<IL, T>, T, std::nullptr_t>::Expression;
};

template<size_t IL, typename T>
struct field_traits<_Field<_impl::ExteriorDerivative<IL, T> > >
{
private:
	static constexpr size_t NDIMS = field_traits<T>::ndims;
//	static constexpr size_t IL = field_traits<T>::iform;

public:
	static constexpr size_t ndims = IL < NDIMS ? NDIMS : 0;
	static constexpr size_t iform = IL + 1;
	static constexpr bool is_field = field_traits<T>::is_field;

	typedef typename field_traits<T>::value_type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;

};
template<size_t IL, typename T>
struct field_traits<_Field<_impl::CodifferentialDerivative<IL, T> > >
{
private:
	static constexpr size_t NDIMS = field_traits<T>::ndims;
//	static constexpr size_t IL = field_traits<T>::iform;
public:
	static const size_t ndims = IL > 0 ? NDIMS : 0;
	static const size_t iform = IL - 1;
	typedef typename field_traits<T>::value_type value_type;

	typedef typename std::conditional<iform == EDGE || iform == FACE,
			nTuple<value_type, 3>, value_type>::type field_value_type;
};

template<typename T>
inline auto exterior_derivative(T const & f)
DECL_RET_TYPE(( _Field<_impl::ExteriorDerivative<
				field_traits<T >::iform , T >>(f)))

template<typename T>
inline auto codifferential_derivative(T const & f)
DECL_RET_TYPE((_Field< _impl::CodifferentialDerivative<
				field_traits<T>::iform , T >>(f)))

template<typename ... T>
inline auto d(_Field<T...> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )

template<typename ... T>
inline auto delta(_Field<T...> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
/**@}*/

/**
 *  @ingroup diff_calculus
 *  @addtogroup vector_calculus Differential calculus of fields
 *  @{
 *
 *  Pseudo-Signature  			| Semantics
 * -------------------------------|--------------
 * \f$\Omega^{1}\f$=Grad(\f$\Omega^0\f$ )		| Grad
 * \f$\Omega^{0}\f$=Diverge(\f$\Omega^1\f$ )	| Diverge
 * \f$\Omega^{2}\f$=Curl(\f$\Omega^1\f$ )		| Curl
 * \f$\Omega^{1}\f$=Curl(\f$\Omega^2\f$ )		| Curl
 *
 *
 */
template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==VERTEX),
		(exterior_derivative(f)))
;
template<typename ... T>
inline auto grad(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==VOLUME),
		((codifferential_derivative(-f))) )
;
template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		(exterior_derivative(f)))
;
template<typename ...T>
inline auto diverge(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==EDGE),
		(codifferential_derivative(-f)))
;
template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==EDGE),
		(exterior_derivative(f)))
;
template<typename ... T>
inline auto curl(_Field<T...> const & f)
ENABLE_IF_DECL_RET_TYPE((field_traits<_Field<T...>>::iform==FACE),
		((codifferential_derivative(-f))) )
;

namespace _impl
{

template<size_t IL, size_t IR, typename T> struct PartialExteriorDerivative
{
};
template<size_t IL, size_t IR, typename T> struct PartialCodifferentialDerivative
{
};
} //namespace _impl

template<size_t IL, size_t IR, typename T>
struct _Field<_impl::PartialExteriorDerivative<IL, IR, T>> : public Expression<
		_impl::PartialExteriorDerivative<IL, IR, T>, T, std::nullptr_t>
{
	using Expression<_impl::PartialExteriorDerivative<IL, IR, T>, T,
			std::nullptr_t>::Expression;
};

template<size_t IL, size_t IR, typename T>
struct _Field<_impl::PartialCodifferentialDerivative<IL, IR, T>> : public Expression<
		_impl::PartialCodifferentialDerivative<IL, IR, T>, T, std::nullptr_t>
{
	using Expression<_impl::PartialCodifferentialDerivative<IL, IR, T>, T,
			std::nullptr_t>::Expression;
};
template<size_t IL, size_t IR, typename T>
struct field_traits<_Field<_impl::PartialExteriorDerivative<IL, IR, T> > >
{
private:
	static constexpr size_t NDIMS = field_traits<T>::ndims;
//	static constexpr size_t IL = field_traits<T>::iform;

public:
	static constexpr size_t ndims = IR < NDIMS ? NDIMS : 0;
	static constexpr size_t iform = IR + 1;
	static constexpr bool is_field = field_traits<T>::is_field;

	typedef typename field_traits<T>::value_type value_type;

};
template<size_t IL, size_t IR, typename T>
struct field_traits<_Field<_impl::PartialCodifferentialDerivative<IL, IR, T> > >
{
private:
	static constexpr size_t NDIMS = field_traits<T>::ndims;
//	static constexpr size_t IL = field_traits<T>::iform;
public:
	static const size_t ndims = IR > 0 ? NDIMS : 0;
	static const size_t iform = IR - 1;
	typedef typename field_traits<T>::value_type value_type;
};

template<size_t IL, typename T>
inline auto p_exterior_derivative(T const & f)
DECL_RET_TYPE(( _Field<_impl::PartialExteriorDerivative<IL,
				field_traits<T >::iform , T >>(f)))
;
template<size_t IL, typename T>
inline auto p_codifferential_derivative(T const & f)
DECL_RET_TYPE((_Field< _impl::PartialCodifferentialDerivative<
				IL,field_traits<T>::iform , T >>(f)))
;
template<typename T> inline auto curl_pdx(T const & f)
ENABLE_IF_DECL_RET_TYPE(field_traits<T>::iform==EDGE,
		(p_exterior_derivative<0>(f )))
;
template<typename T> inline auto curl_pdy(T const & f)
ENABLE_IF_DECL_RET_TYPE(field_traits<T>::iform==EDGE,
		(p_exterior_derivative<1>(f )))
;
template<typename T> inline auto curl_pdz(T const & f)
ENABLE_IF_DECL_RET_TYPE(field_traits<T>::iform==EDGE,
		(p_exterior_derivative<2>(f )))
;
template<typename T> inline auto curl_pdx(T const & f)
ENABLE_IF_DECL_RET_TYPE(field_traits<T>::iform==FACE,
		(p_codifferential_derivative<0>(f )))
;
template<typename T> inline auto curl_pdy(T const & f)
ENABLE_IF_DECL_RET_TYPE(field_traits<T>::iform==FACE,
		(p_codifferential_derivative<1>(f )))
;
template<typename T> inline auto curl_pdz(T const & f)
ENABLE_IF_DECL_RET_TYPE(field_traits<T>::iform==FACE,
		(p_codifferential_derivative<2>(f )))
;
/** @} */
}
// namespace simpla

#endif /* CALCULUS_H_ */
