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
#include "../utilities/sp_type_traits.h"
#include "../utilities/sp_functional.h"
#include "../utilities/constant_ops.h"
namespace simpla
{
template<unsigned int, typename > class nTuple;
template<typename, typename > class Field;
template<typename, unsigned int> class Domain;

struct HodgeStar
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef Domain<TM, TM::ndims - IFORM> type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef Field<typename domain_traits<TD>::type,
				UniOp<HodgeStar, Field<TD, TExpr>> > type;

	};

};

struct InteriorProduct
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef typename std::conditional<IFORM == 0, Zero,
				Domain<TM, IFORM - 1> >::type type;
	};

	template<typename, typename > struct field_traits;

	template<typename TL, typename TD, typename TExpr>
	struct field_traits<TL, Field<TD, TExpr>>
	{
		typedef typename domain_traits<TD>::type domain_type;

		typedef typename std::conditional<
				std::is_same<Zero, domain_type>::value, Zero,
				Field<domain_type, BiOp<InteriorProduct, TL, Field<TD, TExpr>> >> ::type type;

	};

};

struct Wedge
{
	template<typename, typename > struct domain_traits;

	template<typename TM, unsigned int IL, unsigned int IR> struct domain_traits<
			Domain<TM, IL>, Domain<TM, IR>>
	{
		typedef typename std::conditional<(IL + IR > TM::ndims), Zero,
				Domain<TM, IL + IR> >::type type;
	};

	template<typename, typename > struct field_traits;

	template<typename TDL, typename TL, typename TDR, typename TR>
	struct field_traits<Field<TDL, TL>, Field<TDR, TR>>
	{
		typedef typename domain_traits<TDL, TDR>::type domain_type;

		typedef typename std::conditional<
				std::is_same<Zero, domain_type>::value, Zero,
				Field<domain_type, BiOp<Wedge, Field<TDL, TL>, Field<TDL, TR>> >> ::type type;

	};

};

struct ExteriorDerivative
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef typename std::conditional<(IFORM == 0), Zero,
				Domain<TM, IFORM - 1> >::type type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef Field<typename domain_traits<TD>::type,
				UniOp<ExteriorDerivative, Field<TD, TExpr>> > type;

	};
};

struct CodifferentialDerivative
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef typename std::conditional<(IFORM == 0), Zero,
				Domain<TM, IFORM - 1> >::type type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef Field<typename domain_traits<TD>::type,
				UniOp<CodifferentialDerivative, Field<TD, TExpr>> > type;

	};
};

namespace _impl
{
HAS_MEMBER_FUNCTION(domain);

template<typename TF> typename std::enable_if<
		_impl::has_member_function_domain<TF>::value, typename TF::domain_type>::type get_domain(
		TF const & f)
{
	return std::move(f.domain());
}

template<typename TF> typename std::enable_if<
		!_impl::has_member_function_domain<TF>::value, Identity>::type get_domain(
		TF const & f)
{
	return Identity();
}
}  // namespace _impl
/**  \ingroup  FETL
 *  \brief Bi-operation field expression
 */

template<typename TD, typename TOP, typename TL, typename TR>
struct Field<TD, BiOp<TOP, TL, TR> >
{

public:
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;

	typedef TD domain_type;
	typedef typename domain_type::index_type index_type;
	typedef Field<domain_type, BiOp<TOP, TL, TR> > this_type;

	domain_type domain_;

	Field(TL const & l, TR const & r) :
			domain_(_impl::get_domain(l) & _impl::get_domain(r)), l_(l), r_(r)
	{
	}

	domain_type const & domain() const
	{
		return domain_;
	}

	inline auto operator[](index_type const & s) const
	DECL_RET_TYPE( (domain_.template calculus<TOP>( l_,r_,s)))

}
;

/**  \ingroup  FETL
 *   \brief  Uni-operation field expression
 */
template<typename TD, typename TOP, typename TL>
struct Field<TD, UniOp<TOP, TL> >
{

public:

	typename StorageTraits<TL>::const_reference l_;

	typedef TD domain_type;
	typedef typename domain_type::index_type index_type;
	typedef typename domain_type::coordinates_type coordinates_type;
	typedef Field<domain_type, UniOp<TOP, TL> > this_type;

	domain_type domain_;

	Field(TL const & l) :
			l_(l), domain_(_impl::get_domain(l))
	{
	}

	domain_type const &domain() const
	{
		return domain_;
	}

	inline auto operator[](index_type const &s) const
	DECL_RET_TYPE( (domain_.template calculus<TOP>( l_, s)))

};

template<typename TM, unsigned int IFORM, typename TOP, typename TL, typename TR>
struct can_not_reference<Field<Domain<TM, IFORM>, BiOp<TOP, TL, TR> >>
{
	static constexpr bool value = true;
};

template<typename TM, unsigned int IFORM, typename TOP, typename TL>
struct can_not_reference<Field<Domain<TM, IFORM>, UniOp<TOP, TL> > >
{
	static constexpr bool value = true;
};

//! \defgroup  ExteriorAlgebra Exterior algebra
//! @{

template<typename TD, typename TL>
inline typename HodgeStar::field_traits<Field<TD, TL>>::type hodge_star(
		Field<TD, TL> const & f)
{
	return std::move(typename HodgeStar::field_traits<Field<TD, TL>>::type(f));
}

template<typename TDL, typename TDR, typename TL, typename TR>
inline typename Wedge::field_traits<Field<TDL, TL>, Field<TDR, TR>>::type wedge(
		Field<TDL, TL> const & l, Field<TDR, TR> const & r)
{
	return std::move(
			typename Wedge::field_traits<Field<TDL, TL>, Field<TDR, TR>>::type(
					l, r));
}

template<typename TL, typename TD, typename TR>
inline typename InteriorProduct::field_traits<TL, Field<TD, TR>>::type interior_product(
		TL const & l, Field<TD, TR> const & r)
{
	return std::move(
			typename InteriorProduct::field_traits<TL, Field<TD, TR>>::type(l,
					r));
}

template<typename TD, typename TL, typename ... Others>
inline typename ExteriorDerivative::field_traits<Field<TD, TL>>::type exterior_derivative(
		Field<TD, TL> const & f, Others && ...others)
{
	return std::move(
			typename ExteriorDerivative::field_traits<Field<TD, TL>>::type(f,
					std::forward<Others>(others)...));
}

template<typename TD, typename TL, typename ... Others>
inline typename CodifferentialDerivative::field_traits<Field<TD, TL>>::type codifferential_derivative(
		Field<TD, TL> const & f, Others && ...others)
{
	return std::move(
			typename CodifferentialDerivative::field_traits<Field<TD, TL>>::type(
					f, std::forward<Others>(others)...));
}

template<typename TD, typename TL>
inline auto operator*(Field<TD, TL> const & f)
DECL_RET_TYPE((hodge_star(f)))
;
template<typename TD, typename TL>
inline auto d(Field<TD, TL> const & f)
DECL_RET_TYPE( (exterior_derivative(f)) )
;

template<typename TD, typename TL>
inline auto delta(Field<TD, TL> const & f)
DECL_RET_TYPE( (codifferential_derivative(f)) )
;

template<typename TD, typename TL, typename TR>
inline auto iv(nTuple<TD::NDIMS, TL> const & v, Field<TD, TR> const & f)
DECL_RET_TYPE( (interior_product(v,f)) )
;

template<typename TDL, typename TL, typename TDR, typename TR>
inline auto operator^(Field<TDL, TL> const & lhs, Field<TDR, TR> const & rhs)
DECL_RET_TYPE( (wedge(lhs,rhs)) )
;

//!  @}

//!  \ingroup  FETL
//!  \defgroup  VectorAlgebra Vector algebra
//!  @{
template<typename TD, typename TL, typename TR> inline auto InnerProduct(
		Field<TD, TL> const & lhs, Field<TD, TR> const & rhs)
		DECL_RET_TYPE(wedge (lhs,hodge_star( rhs) ))
;

template<typename TM, typename TL, typename TR> inline auto Dot(
		Field<Domain<TM, EDGE>, TL> const & lhs,
		Field<Domain<TM, FACE>, TR> const & rhs)
		DECL_RET_TYPE(wedge(lhs , rhs ))
;

template<typename TM, typename TL, typename TR> inline auto Dot(
		Field<Domain<TM, FACE>, TL> const & lhs,
		Field<Domain<TM, EDGE>, TR> const & rhs)
		DECL_RET_TYPE(wedge(lhs , rhs ))
;

template<typename TM, typename TL, typename TR> inline auto Cross(
		Field<Domain<TM, EDGE>, TL> const & lhs,
		Field<Domain<TM, EDGE>, TR> const & rhs)
		DECL_RET_TYPE( wedge(lhs , rhs ))
;

template<typename TM, typename TL, typename TR> inline auto Dot(
		nTuple<3, TL> const & v, Field<Domain<TM, EDGE>, TR> const & f)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename TM, typename TL, typename TR> inline auto Dot(
		Field<Domain<TM, EDGE>, TR> const & f, nTuple<3, TL> const & v)
		DECL_RET_TYPE( (interior_product(v, f)))
;

template<typename TM, typename TL, typename TR> inline auto Cross(
		Field<Domain<TM, EDGE>, TR> const & f, nTuple<3, TL> const & v)
		DECL_RET_TYPE( (interior_product(v, hodge_star(f))))
;

template<typename TM, typename TL, typename TR> inline auto Cross(
		Field<Domain<TM, FACE>, TR> const & f, nTuple<3, TL> const & v)
		DECL_RET_TYPE((interior_product(v, f)))
;

template<typename TM, typename TR>
inline auto grad(Field<Domain<TM, VERTEX>, TR> const & f)
DECL_RET_TYPE( ( exterior_derivative(f)))
;

template<typename TM, typename TR>
inline auto diverge(Field<Domain<TM, FACE>, TR> const & f)
DECL_RET_TYPE((exterior_derivative( f)))
;

template<typename TM, typename TR>
inline auto curl(Field<Domain<TM, EDGE>, TR> const & f)
DECL_RET_TYPE((exterior_derivative(f)))
;

template<typename TM, typename TR>
inline auto grad(Field<Domain<TM, VOLUME>, TR> const & f)
DECL_RET_TYPE(-(codifferential(f)))
;

template<typename TM, typename TR>
inline auto diverge(Field<Domain<TM, EDGE>, TR> const & f)
DECL_RET_TYPE(-(codifferential_derivative(f)))
;

template<typename TM, typename TR>
inline auto curl(Field<Domain<TM, FACE>, TR> const & f)
DECL_RET_TYPE(-(codifferential_derivative(f)))
;

//!   @}

//!  \ingroup  FETL
//!  \defgroup  NonstandardOperations Non-standard operations
//!   @{
template<typename TM, typename TR>
inline auto CurlPDX(
		Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(
		Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(
		Field<Domain<TM, EDGE>, TR> const & f)
				DECL_RET_TYPE((exterior_derivative(f,std::integral_constant<unsigned int ,2>())))
;

template<typename TM, typename TR>
inline auto CurlPDX(
		Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,0>())))
;

template<typename TM, typename TR>
inline auto CurlPDY(
		Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,1>())))
;

template<typename TM, typename TR>
inline auto CurlPDZ(
		Field<Domain<TM, FACE>, TR> const & f)
				DECL_RET_TYPE((codifferential_derivative(f,std::integral_constant<unsigned int ,2>())))
;

//template<unsigned int IL, typename TM, unsigned int IR, typename TR>
//inline auto MapTo(
//		Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (Field<Domain<TM,IL>, BiOp<MAPTO,std::integral_constant<unsigned int ,IL>,Field<Domain<TM, IR>, TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
//;
//
//template<unsigned int IL, typename TM, unsigned int IR, typename TR>
//inline auto MapTo(std::integral_constant<unsigned int, IL>,
//		Field<Domain<TM, IR>, TR> const & f)
//				DECL_RET_TYPE( (Field<Domain<TM,IL>, BiOp<MAPTO,std::integral_constant<unsigned int ,IL>,Field<Domain<TM, IR>, TR> > >(std::integral_constant<unsigned int ,IL>(), f)))
//;

//!   @}

//! \ingroup FETL
//! \defgroup   BasicAlgebra Basic algebra
//! @{

template<typename TD, typename TL>
inline Field<TD, UniOp<negate<>, Field<TD, TL> > > operator-(
		Field<TD, TL> const & f)
{
	return std::move(Field<TD, UniOp<negate<>, Field<TD, TL> > >(f));
}

template<typename TD, typename TL>
inline Field<TD, TL> const & operator+(Field<TD, TL> const & f)
{
	return f;
}

//template<typename TD, typename TL>
//inline Field<TD, UniOp<RECIPROCAL, Field<TD, TL> > > reciprocal(
//		Field<TD, TL> const & f)
//{
//	return std::move(Field<TD, UniOp<RECIPROCAL, Field<TD, TL> > >(f));
//}

template<typename TD, typename TR>
inline Field<TD, UniOp<op_real<>, Field<TD, TR> > > real(
		Field<TD, TR> const & f)
{
	return std::move(Field<TD, UniOp<op_real<>, Field<TD, TR> > >(f));
}

template<typename TD, typename TR>
inline Field<TD, UniOp<op_imag<>, Field<TD, TR> > > imag(
		Field<TD, TR> const & f)
{
	return std::move(Field<TD, UniOp<op_imag<>, Field<TD, TR> > >(f));
}

template<typename TD, typename TL, typename TR>
inline Field<TD, BiOp<plus<>, Field<TD, TL>, Field<TD, TR>>> operator+(
		Field<TD, TL> const & lhs, Field<TD, TR> const & rhs)
{
	return std::move(Field<TD, BiOp<plus<>, Field<TD, TL>, Field<TD, TR>>>(lhs,rhs));
}

template<typename TD, typename TL, typename TR>
inline Field<TD, BiOp<minus<>, Field<TD, TL>, Field<TD, TR> >> operator-(
		Field<TD, TL> const & lhs, Field<TD, TR> const & rhs)
{
	return std::move(Field<TD, BiOp<minus<>, Field<TD, TL>, Field<TD, TR>>>(lhs,rhs));
}

template<typename TM, typename TL>
inline Field<Domain<TM, VERTEX>,
		BiOp<plus<>, Field<Domain<TM, VERTEX>, TL>, Real>> operator+(
		Field<Domain<TM, VERTEX>, TL> const & lhs, Real rhs)
{
	return std::move(
			Field<Domain<TM, VERTEX>,
					BiOp<minus<>, Field<Domain<TM, VERTEX>, TL>, Real>>(lhs,
					rhs));
}
template<typename TM, typename TL>
inline Field<Domain<TM, VERTEX>,
		BiOp<minus<>, Field<Domain<TM, VERTEX>, TL>, Real>> operator-(
		Field<Domain<TM, VERTEX>, TL> const & lhs, Real rhs)
{
	return std::move(
			Field<Domain<TM, VERTEX>,
					BiOp<minus<>, Field<Domain<TM, VERTEX>, TL>, Real>>(lhs,
					rhs));
}

template<typename TD, typename TL, typename TR>
inline Field<TD, BiOp<multiplies<>, Field<TD, TL>, TR>> operator*(
		Field<TD, TL> const & lhs, TR rhs)
{
	return std::move(Field<TD, BiOp<multiplies<>, Field<TD, TL>, TR>>(lhs, rhs));
}

template<typename TD, typename TL, typename TR>
inline Field<TD, BiOp<divides<>, Field<TD, TL>, TR>> operator/(
		Field<TD, TL> const & lhs, TR const & rhs)
{
	return std::move(Field<TD, BiOp<divides<>, Field<TD, TL>, TR>>(lhs, rhs));
}

template<typename TD, typename TR>
inline Field<TD, BiOp<multiplies<>, Real, Field<TD, TR> >> operator*(Real lhs,
		Field<TD, TR> const & rhs)
{
	return std::move(Field<TD, BiOp<multiplies<>, Real, Field<TD, TR>>>(lhs,rhs));
}

template<typename TD, typename TR>
inline Field<TD, BiOp<multiplies<>, Complex, Field<TD, TR> >> operator*(Complex lhs,
		Field<TD, TR> const & rhs)
{
	return std::move(Field<TD, BiOp<multiplies<>, Complex, Field<TD, TR>>>(lhs,rhs));
}

//! @}


} // namespace simpla

#endif /* CALCULUS_H_ */
