/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * nTuple.h
 *
 *  created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef INCLUDEnTuple_H_
#define INCLUDEnTuple_H_
#include <utility>
#include <ostream>
#include "primitives.h"
#include "sp_complex.h"
#include "sp_functional.h"
#include "sp_type_traits.h"
#include "sp_integer_sequence.h"
#include "expression_template.h"

namespace simpla
{
/**
 * \brief nTuple :n-tuple
 *
 *   Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a positive
 *      unsigned int   eger. There is also one 0-tuple, an empty sequence. An n-tuple is defined
 *    inductively using the construction of an ordered pair. Tuples are usually
 *    written by listing the elements within parenthesis '( )' and separated by
 *    commas; for example, (2, 7, 4, 1, 7) denotes a 5-tuple.
 *    \note  http://en.wikipedia.org/wiki/Tuple
 *   Implement
 *   template< unsigned int     n,typename T> struct nTuple;
 *   nTuple<5,double> t={1,2,3,4,5};
 *
 *
 *   nTuple<T,N...> primary ntuple
 *   nTuple<Expression<TOP,TExpr>> unary nTuple expression
 *   nTuple<Expression<TOP,TExpr1,TExpr2>> binary nTuple expression
 *
 **/
template<typename, unsigned int...> struct nTuple;
template<typename ...>class Expression;

template<typename TV>
struct nTuple_traits
{
	typedef integer_sequence<unsigned int> dimensions;

	static constexpr unsigned int ndims = 0;

	typedef TV value_type;

};

template<typename TV, unsigned int ...N>
struct nTuple_traits<nTuple<TV, N...> >
{

	typedef typename cat_integer_sequence<integer_sequence<unsigned int, N...>,
			typename nTuple_traits<TV>::dimensions>::type dimensions;

	static constexpr unsigned int ndims = dimensions::size();

	typedef typename nTuple_traits<TV>::value_type value_type;

};

template<typename TOP, typename TL>
struct nTuple_traits<nTuple<Expression<TOP, TL> > >
{
private:
	typedef typename nTuple_traits<TL>::dimensions d_seq_l;
	typedef typename nTuple_traits<TL>::value_type value_type_l;
public:
	typedef d_seq_l dimensions;

	static constexpr unsigned int ndims = dimensions::size();

	typedef decltype(std::declval<TOP>()(get_value(std::declval<value_type_l>() ,0))) value_type;

};
template<typename TOP, typename TL, typename TR>
struct nTuple_traits<nTuple<Expression<TOP, TL, TR>> >
{
private:
	typedef typename nTuple_traits<TL>::dimensions d_seq_l;
	typedef typename nTuple_traits<TR>::dimensions d_seq_r;
	typedef typename nTuple_traits<TL>::value_type value_type_l;
	typedef typename nTuple_traits<TR>::value_type value_type_r;
public:
	typedef d_seq_l dimensions;

	static constexpr unsigned int ndims = dimensions::size();

	typedef decltype(std::declval<TOP>()(get_value(std::declval<value_type_l>(),0),
					get_value(std::declval<value_type_r>(),0))) value_type;

};
//
//template<typename ...> struct make_primary_nTuple;
//
//template<typename T, unsigned int M, unsigned int ...N>
//struct make_primary_nTuple<T, integer_sequence<unsigned int, M, N...>>
//{
//private:
//	typedef T value_type;
//	typedef typename make_primary_nTuple<value_type,
//			integer_sequence<unsigned int, N...> >::type sub_type;
//public:
//	typedef nTuple<sub_type, M> type;
//
//};
//template<typename T, unsigned int M>
//struct make_primary_nTuple<T, integer_sequence<unsigned int, M> >
//{
//private:
//	typedef T value_type;
//public:
//	typedef nTuple<value_type, M> type;
//
//};
//
//template<typename T>
//struct make_primary_nTuple<T, integer_sequence<unsigned int> >
//{
//	typedef T value_type;
//	typedef nTuple<value_type> type;
//
//};
//
//
//template<typename T, unsigned int ... N>
//struct make_primary_nTuple<nTuple<T, N...> >
//{
//private:
//	typedef nTuple<T, N...> ntuple_type;
//	typedef T value_type;
//	typedef integer_sequence<unsigned int, N...> dimensions;
//
//public:
//	typedef typename make_primary_nTuple<value_type, dimensions>::type type;
//
//};

template<typename ...> struct make_ndarray_type;

template<typename T, unsigned int M, unsigned int ...N>
struct make_ndarray_type<T, integer_sequence<unsigned int, M, N...>>
{
	typedef typename make_ndarray_type<T, integer_sequence<unsigned int, N...>>::type sub_type;

	typedef sub_type type[M];
};

template<typename T, unsigned int M>
struct make_ndarray_type<T, integer_sequence<unsigned int, M>>
{
	typedef T type[M];
};

//template<typename T>
//struct make_ndarray_type<T, integer_sequence<unsigned int>>
//{
//	typedef T type;
//};

template<typename T>
struct make_ndarray_type<T, integer_sequence<unsigned int>>
{
	typedef T type;
};

template<typename TR, typename T, unsigned int ... N>
void swap(nTuple<T, N...> & l, TR & r)
{
	seq_for<typename nTuple_traits<nTuple<T, N...>>::dimensions>::eval_ndarray(
			_impl::_swap(), (l), (r));
}

template<typename TR, typename T, unsigned int ... N>
void assign(nTuple<T, N...> & l, TR const& r)
{
	seq_for<typename nTuple_traits<nTuple<T, N...>>::dimensions>::eval_ndarray(
			_impl::_assign(), l, r);
}
//template<typename TR, typename T, unsigned int ... N>
//auto inner_product2(nTuple<T, N...> const & l,
//		TR const& r)->decltype(get_value(l,0)*get_value(r,0))
//{
//	return (seq_reduce<typename nTuple_traits<nTuple<T, N...>>::dimensions>::eval_ndarray(
//			_impl::multiplies(), _impl::plus(), l, r));
//}

template<typename TR, typename T, unsigned int ... N>
auto inner_product(nTuple<T, N...> const & l, TR const& r)
DECL_RET_TYPE((
				seq_reduce<typename nTuple_traits<nTuple<T, N...>>::dimensions
				>::eval_ndarray( _impl::multiplies(),_impl::plus(),l, r)
		))

//template<typename T, unsigned int ... N, typename TI>
//auto get_value(nTuple<T, N...> && r, TI const& s)
//DECL_RET_TYPE2((r[s]))

/// n-dimensional primary type
template<typename T, unsigned int ... N>
struct nTuple
{

	typedef T value_type;

	typedef integer_sequence<unsigned int, N...> dimensions;

	static constexpr unsigned int ndims = dimensions::size();

	typedef nTuple<value_type, N...> this_type;

	typedef typename make_ndarray_type<value_type, dimensions>::type data_type;

	data_type data_;

	template<typename TI>
	inline auto operator[](TI const &i) ->decltype(data_[i])
	{
		return data_[i];
	}
	template<typename TI>
	inline auto operator[](TI const &i) const ->decltype(data_[i])
	{
		return data_[i];
	}

	template<unsigned int ...M>
	inline value_type & operator[](
			integer_sequence<unsigned int, M...> const &i)
	{
		return _get_value(data_, i);
	}

	template<unsigned int ...M>
	inline value_type const & operator[](
			integer_sequence<unsigned int, M...> const &i) const
	{
		return _get_value(data_, i);
	}

	template<typename TD, unsigned int M, unsigned int ... L>
	auto _get_value(TD && v,
			integer_sequence<unsigned int, M, L...>)
					DECL_RET_TYPE((_get_value(get_value(std::forward<T>(v),M), integer_sequence<unsigned int, L...>())))

	template<typename TD>
	T& _get_value(TD & v, integer_sequence<unsigned int>)
	{
		return v;
	}

	template<typename TD>
	T const& _get_value(TD const& v, integer_sequence<unsigned int>)
	{
		return v;
	}

//#pragma warning( disable : 597) //disable warning #597: "operator primary_type()" will not be called for implicit or explicit conversions
//
//	typedef typename make_primary_nTuple<this_type>::type primary_type;
//
//	operator primary_type() const
//	{
//		primary_type res;
//		res = *this;
//		return std::move(res);
//	}
//
//#pragma warning( enable : 597)

	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		seq_for<dimensions>::eval(_impl::_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &rhs)
	{
		seq_for<dimensions>::eval(_impl::plus_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		seq_for<dimensions>::eval(_impl::minus_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		seq_for<dimensions>::eval(_impl::multiplies_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		seq_for<dimensions>::eval(_impl::divides_assign(), data_, rhs);
		return (*this);
	}

//	template<unsigned int NR, typename TR>
//	void operator*(nTuple<NR, TR> const & rhs) = delete;
//
//	template<unsigned int NR, typename TR>
//	void operator/(nTuple<NR, TR> const & rhs) = delete;

};

template<typename ... T>
struct nTuple<Expression<T...>> : public Expression<T...>
{
	typedef nTuple<Expression<T...>> this_type;
	typedef typename nTuple_traits<this_type>::value_type value_type;
	typedef typename nTuple_traits<this_type>::dimensions dimensions;
//	typedef typename make_primary_nTuple<value_type, dimensions>::type primary_type;

//	operator bool() const
//	{
//		return seq_reduce<dimensions>::eval(_impl::logical_and(), *this);
//	}

	using Expression<T...>::Expression;
};

template<typename T, unsigned int ...N>
std::ostream &operator<<(std::ostream & os, nTuple<T, N...> const & v)
{
	os << "UNIMPLEMENT" << std::endl;
	return os;
}

typedef nTuple<Real, 3> Vec3;

typedef nTuple<Real, 3> IVec3;

typedef nTuple<Integral, 3> RVec3;

typedef nTuple<Complex, 3> CVec3;

/// Eigen style

template<typename T, unsigned int N> using Vector=nTuple<T,N>;

template<typename T, unsigned int M, unsigned int N> using Matrix=nTuple<T,M,N >;

template<typename T, unsigned int ... N> using Tensor=nTuple<T,N...>;

template<typename T> inline auto determinant(
		Matrix<T, 3, 3> const & m)
				DECL_RET_TYPE(( m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
								* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
								* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
						)

				)

template<typename T> inline auto determinant(
		Matrix<T, 4, 4> const & m)
		DECL_RET_TYPE((//
				m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]//
				- m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3]//
				* m[2][2] * m[3][0] + m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1]//
				* m[1][2] * m[2][3] * m[3][0] - m[0][3] * m[1][2] * m[2][0] * m[3][1]//
				+ m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2]//
				* m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] - m[0][2] * m[1][0]//
				* m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] + m[0][3]//
				* m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2]//
				- m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1]//
				* m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1]//
				* m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1]//
				* m[1][2] * m[2][0] * m[3][3] + m[0][2] * m[1][0] * m[2][1] * m[3][3]//
				- m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2]//
				* m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3]//
		))

template<typename T, unsigned int ...N> auto abs(nTuple<T, N...> const & m)
DECL_RET_TYPE((std::sqrt(std::abs(dot(m, m)))))

template<unsigned int N, typename T> inline auto NProduct(
		nTuple<T, N> const & v)
		->decltype(v[0]*v[1])
{
	decltype(v[0]*v[1]) res;
	res = 1;
	for (unsigned int i = 0; i < N; ++i)
	{
		res *= v[i];
	}
	return res;

}
template<unsigned int N, typename T> inline auto NSum(nTuple<T, N> const & v)
->decltype(v[0]+v[1])
{
	decltype(v[0]+v[1]) res;
	res = 0;
	for (unsigned int i = 0; i < N; ++i)
	{
		res += v[i];
	}
	return res;
}

template<typename T1, unsigned int ... N, typename TR>
inline auto dot(nTuple<T1, N...> const &l, TR const &r)
DECL_RET_TYPE((inner_product(l,r)))

template<typename T1, unsigned int ... N1, typename T2, unsigned int ... N2> inline auto cross(
		nTuple<T1, N1...> const & l, nTuple<T2, N2...> const & r)
		->nTuple<decltype(get_value(l,0)*get_value(r,0)),3>
{
	nTuple<decltype(get_value(l,0)*get_value(r,0)), 3> res = { l[1] * r[2]
			- l[2] * r[1], l[2] * get_value(r, 0) - get_value(l, 0) * r[2],
			get_value(l, 0) * r[1] - l[1] * get_value(r, 0) };
	return std::move(res);
}
//template<unsigned int ndims, typename TExpr>
//auto operator >>(nTuple<ndims, TExpr> const & v,
//		unsigned int n)-> nTuple<ndims,decltype(v[0] >> n )>
//{
//	nTuple<ndims, decltype(v[0] >> n )> res;
//	for (unsigned int i = 0; i < ndims; ++i)
//	{
//		res[i] = v[i] >> n;
//	}
//	return res;
//}
//
//template<unsigned int ndims, typename TExpr>
//auto operator <<(nTuple<ndims, TExpr> const & v,
//		unsigned int n)-> nTuple<ndims,decltype(v[0] << n )>
//{
//	nTuple<ndims, decltype(v[0] >> n )> res;
//	for (unsigned int i = 0; i < ndims; ++i)
//	{
//		res[i] = v[i] << n;
//	}
//	return res;
//}

template<typename TP, TP ...J>
nTuple<TP, (sizeof...(J)) + 1> makenTuple(integer_sequence<TP, J...>)
{
	constexpr unsigned int N = 1 + sizeof...(J);
	typedef nTuple<TP, N> type;
	type res;
//
//	seq_for<integer_sequence<unsigned int, N>>::eval_multi_parameter(
//			[](type & r,unsigned int i)
//			{
//				r[i+1]=i;
//			}, res);
	return std::move(res);
}

//template<typename TP>
//unsigned int makenTuple(integer_sequence<TP>)
//{
//	return 0;
//}

#define _SP_DEFINE_nTuple_EXPR_BINARY_RIGHT_OPERATOR(_OP_, _NAME_)                                                  \
	template<typename T1,unsigned int ...N1,typename  T2> auto operator _OP_(nTuple<T1,N1...> const & l,T2 const &r)  \
	DECL_RET_TYPE2((nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r)))                  \


#define _SP_DEFINE_nTuple_EXPR_BINARY_OPERATOR(_OP_,_NAME_)                                                  \
	template<typename T1,unsigned int ...N1,typename  T2> auto operator _OP_(nTuple<T1,N1...> const & l,T2 const &r)  \
	DECL_RET_TYPE2((nTuple<Expression<_impl::_NAME_,nTuple<T1,N1...>,T2>>(l,r)))                  \
	template< typename T1,typename T2 ,unsigned int ...N2> auto operator _OP_(T1 const & l, nTuple< T2,N2...>const &r)                    \
	DECL_RET_TYPE2((nTuple<Expression< _impl::_NAME_,T1,nTuple< T2,N2...>>>(l,r)))                  \
	template< typename T1,unsigned int ... N1,typename T2 ,unsigned int ...N2>  \
	auto operator _OP_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
	DECL_RET_TYPE2((nTuple<Expression< _impl::_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r)))                  \


#define _SP_DEFINE_nTuple_EXPR_UNARY_OPERATOR(_OP_,_NAME_)                           \
		template<typename T,unsigned int ...N> auto operator _OP_(nTuple<T,N...> const &l)  \
		DECL_RET_TYPE2((nTuple<Expression<_impl::_NAME_,nTuple<T,N...> >>(l)))   \

#define _SP_DEFINE_nTuple_EXPR_BINARY_FUNCTION(_NAME_)                                                  \
			template<typename T1,unsigned int ...N1,typename  T2> auto   _NAME_(nTuple<T1,N1...> const & l,T2 const &r)  \
			DECL_RET_TYPE2((nTuple<Expression<_impl::_##_NAME_,nTuple<T1,N1...>,T2>>(l,r)))                  \
			template< typename T1,typename T2,unsigned int ...N2> auto _NAME_(T1 const & l, nTuple< T2,N2...>const &r)                    \
			DECL_RET_TYPE2((nTuple<Expression< _impl::_##_NAME_,T1,nTuple< T2,N2...>>>(l,r)))                  \
			template< typename T1,unsigned int ... N1,typename T2,unsigned int  ...N2> \
			auto _NAME_(nTuple< T1,N1...> const & l,nTuple< T2,N2...>  const &r)                    \
			DECL_RET_TYPE2((nTuple<Expression< _impl::_##_NAME_,nTuple< T1,N1...>,nTuple< T2,N2...>>>(l,r)))                  \


#define _SP_DEFINE_nTuple_EXPR_UNARY_FUNCTION( _NAME_)                           \
		template<typename T,unsigned int ...N> auto   _NAME_(nTuple<T,N ...> const &r)  \
		DECL_RET_TYPE2((nTuple<Expression<_impl::_##_NAME_,nTuple<T,N...>>>(r)))   \

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(nTuple)
}
//namespace simpla

#endif  // INCLUDEnTuple_H_
