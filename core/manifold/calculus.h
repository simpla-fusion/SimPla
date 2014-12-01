/*
 * calculus.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_
#include <type_traits>

#include "../utilities/constant_ops.h"
#include "../utilities/primitives.h"
#include "../utilities/ntuple.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/sp_functional.h"
#include "../utilities/expression_template.h"
#include "domain.h"

namespace simpla
{
template<typename ... > class _Field;
template<typename, size_t> class Domain;
template<typename, typename, typename > class Expression;

/// \defgroup  ExteriorAlgebra Exterior algebra
/// @{
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
template<size_t IL, typename T> struct ExteriorDerivative
{
};
template<size_t IL, typename T> struct CodifferentialDerivative
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

	typedef typename std::result_of<_impl::multiplies(l_type, r_type)>::type value_type;

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

	typedef typename std::result_of<_impl::multiplies(l_type, r_type)>::type value_type;

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

template<typename T>
inline auto exterior_derivative(T const & f)
DECL_RET_TYPE(( _Field<_impl::ExteriorDerivative<
				field_traits<T >::iform , T >>(f)))

template<typename T>
inline auto codifferential_derivative(T const & f)
DECL_RET_TYPE((_Field< _impl::CodifferentialDerivative<
				field_traits<T>::iform , T >>(f)))

template<typename ...T>
inline auto operator*(_Field<T...> const & f)
DECL_RET_TYPE((hodge_star(f)))

template<typename ... T>
inline auto d(_Field<T...> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )

template<typename ... T>
inline auto delta(_Field<T...> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )

template<size_t ndims, typename TL, typename ...T>
inline auto iv(nTuple<TL, ndims> const & v, _Field<T...> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )

template<typename ...T1, typename ... T2>
inline auto operator^(_Field<T1...> const & lhs, _Field<T2...> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )

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

//template<typename ...T>
//inline auto grad(_Field<T...> const & f)
//DECL_RET_TYPE( ( exterior_derivative(f)))
//;

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
///   @}

///  \ingroup  FETL
///  \defgroup  NonstandardOperations Non-standard operations
///   @{

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

///   @}

}
// namespace simpla

#endif /* CALCULUS_H_ */
