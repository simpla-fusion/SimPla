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
//#include "ntuple_noet.h"
#include "primitives.h"
#include "sp_complex.h"
#include "sp_functional.h"
#include "sp_type_traits.h"

namespace simpla
{

namespace _ntuple_impl
{

template<unsigned int M, typename TL, typename TR> struct _swap
{
	static inline void eval(TL & l, TR &r)
	{
		std::swap(l[M - 1], r[M - 1]);
		_swap<M - 1, TL, TR>::calculus(l, r);
	}
};
template<typename TL, typename TR> struct _swap<1, TL, TR>
{
	static inline void eval(TL & l, TR &r)
	{
		std::swap(l[0], r[0]);
	}
};
template<unsigned int N, typename TL, typename TR>
void swap(TL & l, TR & r)
{
	_swap<N, TL, TR>::eval(l, r);
}

template<unsigned int M> struct _assign
{
	template<typename TL, typename TR>
	static inline void eval(TL & l, TR const &r)
	{
		l[M - 1] = get_value(r, M - 1);
		_assign<M - 1>::eval(l, r);
	}
	template<typename TFun, typename TL, typename TR>
	static inline void eval(TFun const & fun, TL & l, TR const &r)
	{
		l[M - 1] = fun(l[M - 1], get_value(r, M - 1));
		_assign<M - 1>::eval(fun, l, r);
	}

};
template<> struct _assign<1>
{
	template<typename TL, typename TR>
	static inline void eval(TL & l, TR const &r)
	{
		l[0] = get_value(r, 0);
	}
	template<typename TFun, typename TL, typename TR>
	static inline void eval(TFun const & fun, TL & l, TR const &r)
	{
		auto & v = l[0];
		v = fun(v, get_value(r, 0));
	}
};

template<unsigned int N, typename TL, typename TR>
void assign(TL & l, TR const & r)
{
	_assign<N>::eval(l, r);
}
template<unsigned int N, typename TFun, typename TL, typename TR>
void assign(TFun const & fun, TL & l, TR const & r)
{
	_assign<N>::eval(fun, l, r);
}
template<unsigned int N>
struct _reduce
{
	template<typename TPlus, typename TMultiplies, typename TL, typename TR>
	static inline auto eval(TPlus const & plus_op, TMultiplies const &multi_op,
			TL const & l,
			TR const &r)
					DECL_RET_TYPE ((plus_op(multi_op(get_value(l,N - 1),get_value( r,N - 1)), _reduce<N - 1>::eval(plus_op, multi_op, l, r))))

};

template<>
struct _reduce<1>
{
	template<typename TPlus, typename TMultiplies, typename TL, typename TR>
	static inline auto eval(TPlus const & plus_op, TMultiplies const &multi_op,
			TL const & l, TR const &r)
			DECL_RET_TYPE ((multi_op(get_value(l,0),get_value( r,0))))

};
template<unsigned int N, typename TPlus, typename TMultiplies, typename TL,
		typename TR>
auto reduce(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l,
		TR const & r)
		DECL_RET_TYPE (_reduce<N>::eval(plus_op, multi_op, l, r))

}
// namespace ntuple_impl
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
template<unsigned int N, typename ...T> struct nTuple;

template<unsigned int N, typename ...T>
auto get_value(nTuple<N, T...> const & r, size_t s)
DECL_RET_TYPE((r[s]))

template<unsigned int N, typename T>
struct nTuple<N, T>
{
	static const unsigned int DIMENSION = N;
	typedef nTuple<DIMENSION, T> this_type;
	typedef T value_type;

	value_type data_[N];

	inline value_type &
	operator[](size_t i)
	{
		return (data_[i]);
	}

	inline value_type const&
	operator[](size_t i) const
	{
		return (data_[i]);
	}
	inline void swap(this_type & rhs)
	{
		_ntuple_impl::swap<N>(*this, rhs);
	}

	template<typename TR>
	inline bool operator ==(TR const &rhs) const
	{
		return _ntuple_impl::reduce<N>(_impl::logical_and(), _impl::equal_to(),
				*this, rhs);
	}

	template<typename TExpr>
	inline bool operator !=(nTuple<DIMENSION, TExpr> const &rhs) const
	{
		return _ntuple_impl::reduce<N>(_impl::logical_and(),
				_impl::not_equal_to(), *this, rhs);
	}
	template<typename TR>
	inline operator nTuple<N,TR>() const
	{
		nTuple<N, TR> res;

		_ntuple_impl::assign<N>(_impl::binary_right(), res, data_);

		return (res);
	}

	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		_ntuple_impl::assign<N>(_impl::binary_right(), data_, rhs);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR rhs[])
	{
		_ntuple_impl::assign<N>(_impl::binary_right(), data_, rhs);

		return (*this);
	}
	template<typename TR> inline this_type &
	operator =(nTuple<N, TR> const &rhs)
	{
		_ntuple_impl::assign<N>(_impl::binary_right(), data_, rhs);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(_impl::plus(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(_impl::minus(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(_impl::multiplies(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(_impl::divides(), data_, rhs);
		return (*this);
	}

	template<unsigned int NR, typename TR>
	void operator*(nTuple<NR, TR> const & rhs) = delete;

	template<unsigned int NR, typename TR>
	void operator/(nTuple<NR, TR> const & rhs) = delete;

};

template<unsigned int N, typename TL>
struct can_not_reference<nTuple<N, TL>>
{
	static constexpr bool value = false;
};

template<unsigned int N, typename T> using Matrix=nTuple<N,nTuple<N,T>>;

template<typename T>
auto make_ntuple(T v0)
DECL_RET_TYPE(v0)
;

template<typename T>
nTuple<2, T> make_ntuple(T v0, T v1)
{
	return std::move(nTuple<2, T>(
	{ v0, v1 }));
}

template<typename T>
auto make_ntuple(T v0, T v1, T v2)
DECL_RET_TYPE((nTuple<3,T>(
						{	v0,v1,v2})))
;

template<typename T>
auto make_ntuple(T v0, T v1, T v2, T v3)
DECL_RET_TYPE((nTuple<3,T>(
						{	v0,v1,v2,v3})))
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
template<typename TV>
struct nTupleTraits
{
	static constexpr unsigned int NDIMS = 0;
	static constexpr unsigned int DIMENSION = 1;
	typedef TV value_type;
	typedef value_type element_type;

	template<typename TI>
	static void get_dimensions(TI *)
	{
	}

};

template<unsigned int N, typename TV>
struct nTupleTraits<nTuple<N, TV>>
{
	static constexpr unsigned int NDIMS = nTupleTraits<TV>::NDIMS + 1;
	static constexpr unsigned int DIMENSION = N;

	typedef TV value_type;

	typedef typename nTupleTraits<TV>::element_type element_type;

	template<typename TI>
	static void get_dimensions(TI* dims)
	{
		if (dims != nullptr)
		{
			dims[NDIMS - 1] = DIMENSION;
			nTupleTraits<TV>::get_dimensions(dims);
		}
	}

};
//
//template<unsigned int N, class T>
//class is_indexable<nTuple<N, T> >
//{
//public:
//	static const bool value = true;
//
//};

//template<typename TL, typename TR>
//inline auto dot(nTuple<2, TL> const &l, nTuple<2, TR> const &r)
//DECL_RET_TYPE((l[0]*r[0]+l[1]*r[1] ))
//
//template<typename TL, typename TR>
//inline auto dot(nTuple<3, TL> const &l, nTuple<3, TR> const &r)
//DECL_RET_TYPE((l[0]*r[0]+l[1]*r[1]+l[2]*r[2]))
//
//template<typename TL, typename TR>
//inline auto dot(nTuple<4, TL> const &l, nTuple<4, TR> const &r)
//DECL_RET_TYPE((l[0]*r[0]+l[1]*r[1]+l[2]*r[2]+l[3]*r[3]))
//
template<typename T> inline auto determinant(
		nTuple<3, nTuple<3, T> > const & m)
				DECL_RET_TYPE(( m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
								* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
								* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
						)

				)

template<typename T> inline auto determinant(
		nTuple<4, nTuple<4, T> > const & m) DECL_RET_TYPE(
				(//
				m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1]
				* m[3][0] - m[0][3] * m[1][1] * m[2][2] * m[3][0]
				+ m[0][1] * m[1][3]//
				* m[2][2] * m[3][0] + m[0][2] * m[1][1] * m[2][3] * m[3][0]
				- m[0][1]//
				* m[1][2] * m[2][3] * m[3][0]
				- m[0][3] * m[1][2] * m[2][0] * m[3][1]//
				+ m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2]//
				* m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1]
				- m[0][2] * m[1][0]//
				* m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1]
				+ m[0][3]//
				* m[1][1] * m[2][0] * m[3][2]
				- m[0][1] * m[1][3] * m[2][0] * m[3][2]//
				- m[0][3] * m[1][0] * m[2][1] * m[3][2]
				+ m[0][0] * m[1][3] * m[2][1]//
				* m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2]
				- m[0][0] * m[1][1]//
				* m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3]
				+ m[0][1]//
				* m[1][2] * m[2][0] * m[3][3]
				+ m[0][2] * m[1][0] * m[2][1] * m[3][3]//
				- m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2]//
				* m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3]//
		))

template<unsigned int N, typename T> auto abs(nTuple<N, T> const & m)
DECL_RET_TYPE((std::sqrt(std::abs(dot(m, m)))))

template<unsigned int N, typename T> inline auto NProduct(
		nTuple<N, T> const & v)
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
template<unsigned int N, typename T> inline auto NSum(nTuple<N, T> const & v)
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
//template<unsigned int N, unsigned int M, typename T> Real abs(
//		nTuple<N, nTuple<M, T>> const & m)
//{
//	T res = 0.0;
//	for (unsigned int i = 0; i < N; ++i)
//		for (unsigned int j = 0; j < M; ++j)
//		{
//			res += m[i][j] * m[i][j];
//		}
//
//	return (std::sqrt(abs(res)));
//}

//template<unsigned int NDIMS, typename TExpr>
//auto operator >>(nTuple<NDIMS, TExpr> const & v,
//		unsigned int n)-> nTuple<NDIMS,decltype(v[0] >> n )>
//{
//	nTuple<NDIMS, decltype(v[0] >> n )> res;
//	for (unsigned int i = 0; i < NDIMS; ++i)
//	{
//		res[i] = v[i] >> n;
//	}
//	return res;
//}
//
//template<unsigned int NDIMS, typename TExpr>
//auto operator <<(nTuple<NDIMS, TExpr> const & v,
//		unsigned int n)-> nTuple<NDIMS,decltype(v[0] << n )>
//{
//	nTuple<NDIMS, decltype(v[0] >> n )> res;
//	for (unsigned int i = 0; i < NDIMS; ++i)
//	{
//		res[i] = v[i] << n;
//	}
//	return res;
//}

}
//namespace simpla

#include "ntuple_et.h"
#endif  // INCLUDE_NTUPLE_H_
