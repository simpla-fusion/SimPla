/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * nTuple.h
 *
 *  created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef INCLUDE_NTUPLE_H_
#define INCLUDE_NTUPLE_H_
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
 *   nTuple<N,T> primary ntuple
 *   nTuple<N,TOP,TExpr> unary nTuple expression
 *   nTuple<N,TOP,TExpr1,TExpr2> binary nTuple expression
 *
 **/
template<typename ...> struct _nTuple;

template<typename TV>
struct nTuple_traits
{
	typedef integer_sequence<unsigned int> dimensions;

	static constexpr unsigned int ndims = 0;

	typedef TV value_type;

};

template<typename TV, unsigned int ...N>
struct nTuple_traits<_nTuple<TV, integer_sequence<unsigned int, N...>> >
{

	typedef typename cat_integer_sequence<integer_sequence<unsigned int, N...>,
			typename nTuple_traits<TV>::dimensions>::type dimensions;

	static constexpr unsigned int ndims = dimensions::size();

	typedef typename nTuple_traits<TV>::value_type value_type;

};
template<typename ...>class Expression;

template<typename TOP, typename TL>
struct nTuple_traits<_nTuple<Expression<TOP, TL> > >
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
struct nTuple_traits<_nTuple<Expression<TOP, TL, TR>> >
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

template<typename ...> struct make_primary_ntuple;

template<typename T, unsigned int M, unsigned int ...N>
struct make_primary_ntuple<T, integer_sequence<unsigned int, M, N...>>
{
private:
	typedef T value_type;
	typedef typename make_primary_ntuple<value_type,
			integer_sequence<unsigned int, N...> >::type sub_type;
public:
	typedef _nTuple<sub_type, integer_sequence<unsigned int, M>> type;

};
template<typename T, unsigned int M>
struct make_primary_ntuple<T, integer_sequence<unsigned int, M>>
{
private:
	typedef T value_type;
public:
	typedef _nTuple<value_type, integer_sequence<unsigned int, M>> type;

};

template<typename T>
struct make_primary_ntuple<T, integer_sequence<unsigned int>>
{
	typedef T value_type;
	typedef _nTuple<value_type, integer_sequence<unsigned int>> type;

};
template<typename ...T>
struct make_primary_ntuple<_nTuple<T...> >
{
private:
	typedef _nTuple<T...> ntuple_type;
	typedef typename nTuple_traits<ntuple_type>::value_type value_type;
	typedef typename nTuple_traits<ntuple_type>::dimensions dimensions;

public:
	typedef typename make_primary_ntuple<value_type, dimensions>::type type;

};

template<typename ...> struct make_ndarray_type;
template<typename T, unsigned int M, unsigned int ...N>
struct make_ndarray_type<T, integer_sequence<unsigned int, M, N...> >
{
	typedef typename make_ndarray_type<T, integer_sequence<unsigned int, N...>>::type sub_type;

	typedef sub_type type[M];
};

template<typename T, unsigned int M>
struct make_ndarray_type<T, integer_sequence<unsigned int, M> >
{
	typedef T type[M];
};

template<typename T>
struct make_ndarray_type<T, integer_sequence<unsigned int> >
{
	typedef T type;
};

template<typename TR, typename ...T>
void swap(_nTuple<T...> & l, TR & r)
{
	seq_for<typename nTuple_traits<_nTuple<T...>>::dimensions>::eval_ndarray(
			_impl::_swap(), (l), (r));
}

template<typename TR, typename ...T>
void assign(_nTuple<T...> & l, TR const& r)
{
	seq_for<typename nTuple_traits<_nTuple<T...>>::dimensions>::eval_ndarray(
			_impl::_assign(), l, r);
}
template<typename TR, typename ...T>
auto inner_product2(_nTuple<T...> const & l,
		TR const& r)->decltype(get_value(l,0)*get_value(r,0))
{
	return (seq_reduce<typename nTuple_traits<_nTuple<T...>>::dimensions>::eval_ndarray(
			_impl::multiplies(), _impl::plus(), l, r));
}

template<typename TR, typename ...T>
auto inner_product(_nTuple<T...> const & l, TR const& r)
DECL_RET_TYPE((
				seq_reduce<typename nTuple_traits<_nTuple<T...>>::dimensions
				>::eval_ndarray( _impl::multiplies(),_impl::plus(),l, r)
		))

template<typename ...T, typename TI>
auto get_value(_nTuple<T...> && r, TI const& s)
DECL_RET_TYPE((r[s]))

/// n-dimensional primary type
template<typename T, unsigned int ... N>
struct _nTuple<T, integer_sequence<unsigned int, N...> >
{

	typedef T value_type;
	typedef integer_sequence<unsigned int, N...> dimensions;

	typedef _nTuple<value_type, dimensions> this_type;

	typedef typename make_ndarray_type<value_type, dimensions>::type data_type;

	static constexpr unsigned int ndims = dimensions::size();

	typedef typename std::conditional<ndims <= 1, size_t,
			_nTuple<size_t, integer_sequence<unsigned int, ndims> > >::type index_type;

	data_type data_;

	inline value_type & operator[](index_type const &i)
	{
		return (get_value(data_, i));
	}

	inline value_type const &operator[](index_type const& i) const
	{
		return (get_value(data_, i));
	}

//	template<typename TR>
//	inline bool operator ==(TR const &rhs) const
//	{
//		return seq_reduce<dimensions>::eval_ndarray(_impl::equal_to(),
//				_impl::logical_and(), *this, rhs);
//	}
//
//	template<typename TR>
//	inline bool operator !=(TR const &rhs) const
//	{
//		return !(*this == rhs);
//	}

#pragma warning( disable : 597) //disable warning #597: "operator primary_type()" will not be called for implicit or explicit conversions

	typedef typename make_primary_ntuple<this_type>::type primary_type;

	operator primary_type() const
	{
		primary_type res;
		res = *this;
		return std::move(res);
	}

#pragma warning( enable : 597)

	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		seq_for<dimensions>::eval_ndarray(_impl::_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &rhs)
	{
		seq_for<dimensions>::eval_ndarray(_impl::plus_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		seq_for<dimensions>::eval_ndarray(_impl::minus_assign(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		seq_for<dimensions>::eval_ndarray(_impl::multiplies_assign(), data_,
				rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		seq_for<dimensions>::eval_ndarray(_impl::divides_assign(), data_, rhs);
		return (*this);
	}

//	template<unsigned int NR, typename TR>
//	void operator*(nTuple<NR, TR> const & rhs) = delete;
//
//	template<unsigned int NR, typename TR>
//	void operator/(nTuple<NR, TR> const & rhs) = delete;

};

template<typename ... T>
struct _nTuple<Expression<T...>> : public Expression<T...>
{
	typedef _nTuple<Expression<T...>> this_type;
	typedef typename nTuple_traits<this_type>::value_type value_type;
	typedef typename nTuple_traits<this_type>::dimensions dimensions;
	typedef typename make_primary_ntuple<value_type, dimensions>::type primary_type;

	operator primary_type() const
	{
		primary_type res;
		res = *this;
		return std::move(res);
	}

	operator bool() const
	{
		return seq_reduce<dimensions>::eval_ndarray(_impl::logical_and(), *this);
	}

	using Expression<T...>::Expression;
};

template<typename ...T>
std::ostream &operator<<(std::ostream & os, _nTuple<T...> const & v)
{
	os << "UNIMPLEMENT" << std::endl;
	return os;
}

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_nTuple)

template<unsigned int N, typename T> using nTuple=_nTuple<T,integer_sequence<unsigned int,N >>;

typedef _nTuple<Real, integer_sequence<unsigned int, 3>> Vec3;

typedef _nTuple<Real, integer_sequence<unsigned int, 3>> IVec3;

typedef _nTuple<Integral, integer_sequence<unsigned int, 3>> RVec3;

typedef _nTuple<Complex, integer_sequence<unsigned int, 3>> CVec3;

/// Eigen style

template<typename T, unsigned int N> using Vector=_nTuple<T,integer_sequence<unsigned int,N >>;

template<typename T, unsigned int ... N> using Matrix=typename make_primary_ntuple<_nTuple<T,integer_sequence<unsigned int,N...>>>::type;

template<typename T, unsigned int ...N> using Tensor=_nTuple<T,integer_sequence<unsigned int,N...>>;

//
//template<typename T>
//auto make_ntuple(T v0)
//DECL_RET_TYPE(v0)
//;
//
//template<typename T>
//nTuple<2, T> make_ntuple(T v0, T v1)
//{
//	return std::move(nTuple<2, T>(
//	{ v0, v1 }));
//}
//
//template<typename T>
//auto make_ntuple(T v0, T v1, T v2)
//DECL_RET_TYPE((nTuple<3,T>(
//						{	v0,v1,v2})))
//;
//
//template<typename T>
//auto make_ntuple(T v0, T v1, T v2, T v3)
//DECL_RET_TYPE((nTuple<3,T>(
//						{	v0,v1,v2,v3})))
;

//template<typename TV>
//struct is_nTuple
//{
//	static constexpr bool value = false;
//
//};
//
//template<unsigned int N, typename TV>
//struct is_nTuple<nTuple<N, TV>>
//{
//	static constexpr bool value = true;
//
//};
//
//template<unsigned int N, typename TE>
//struct is_primitive<nTuple<N, TE> >
//{
//	static constexpr bool value = is_arithmetic_scalar<TE>::value;
//};
//

//
//template<unsigned int N, class T>
//class is_indexable<nTuple<N, T> >
//{
//public:
//	static const bool value = true;
//
//};

template<typename T> inline auto determinant(
		_nTuple<_nTuple<T, integer_sequence<unsigned int, 3>>,
				integer_sequence<unsigned int, 3> > const & m)
				DECL_RET_TYPE(( m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
								* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
								* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
						)

				)

template<typename T> inline auto determinant(
		_nTuple<_nTuple<T, integer_sequence<unsigned int, 4>>,
				integer_sequence<unsigned int, 4> > const & m)
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

template<typename T, unsigned int ...N> auto abs(
		_nTuple<T, integer_sequence<unsigned int, N...>> const & m)
		DECL_RET_TYPE((std::sqrt(std::abs(dot(m, m)))))

template<unsigned int N, typename T> inline auto NProduct(
		_nTuple<T, integer_sequence<unsigned int, N> > const & v)
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
template<unsigned int N, typename T> inline auto NSum(
		_nTuple<T, integer_sequence<unsigned int, N>> const & v)
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

template<typename ... T1, typename TR>
inline auto dot(_nTuple<T1...> const &l, TR const &r)
DECL_RET_TYPE((inner_product(l,r)))

template<typename ... T1, typename ...T2> inline auto cross(
		_nTuple<T1...> const & l, _nTuple<T2...> const & r)
		->nTuple<3,decltype(get_value(l,0)*get_value(r,0))>
{
	nTuple<3, decltype(get_value(l,0)*get_value(r,0))> res = { l[1] * r[2]
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
_nTuple<TP, integer_sequence<unsigned int, sizeof...(J)>> make_ntuple(
		integer_sequence<TP, J...>)
{
	constexpr unsigned int N = 1 + sizeof...(J);
	typedef _nTuple<TP, integer_sequence<unsigned int, N>> type;
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
//unsigned int make_ntuple(integer_sequence<TP>)
//{
//	return 0;
//}
}
//namespace simpla

#endif  // INCLUDE_NTUPLE_H_
