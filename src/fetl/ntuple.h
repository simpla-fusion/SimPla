/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * nTuple.h
 *
 *  Created on: Jan 27, 2010
 *      Author: yuzhi
 */

#ifndef INCLUDE_NTUPLE_H_
#define INCLUDE_NTUPLE_H_

#include <complex>
#include <cstddef>
#include <initializer_list>
#include "complex_ops.h"
#include "primitives.h"

#include "ntuple_noet.h"

#include "../utilities/type_utilites.h"

namespace simpla
{

/**
 *  nTuple :n-tuple
 *  @Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a positive
 *    integer. There is also one 0-tuple, an empty sequence. An n-tuple is defined
 *    inductively using the construction of an ordered pair. Tuples are usually
 *    written by listing the elements within parenthesis '( )' and separated by
 *    commas; for example, (2, 7, 4, 1, 7) denotes a 5-tuple.
 *    @url: http://en.wikipedia.org/wiki/Tuple
 *  @Implement
 *   template<int n,typename T> struct nTuple;
 *   nTuple<5,double> t={1,2,3,4,5};
 *
 *	@ingroup ntuple
 * */

template<int N, typename T> struct nTuple;
template<int N, typename T> using Matrix=nTuple<N,nTuple<N,T>>;

namespace _ntuple_impl
{

template<int M, typename TL, typename TR> struct _swap
{
	static inline void eval(TL & l, TR &r)
	{
		std::swap(l[M - 1], r[M - 1]);
		_swap<M - 1, TL, TR>::eval(l, r);
	}
};
template<typename TL, typename TR> struct _swap<1, TL, TR>
{
	static inline void eval(TL & l, TR &r)
	{
		std::swap(l[0], r[0]);
	}
};
template<int N, typename TL, typename TR>
void swap(TL & l, TR & r)
{
	_swap<N, TL, TR>::eval(l, r);
}

struct binary_right
{
	template<typename TL, typename TR>
	TR const &operator()(TL const &, TR const & r) const
	{
		return r;
	}
};
template<int M> struct _assign
{
	template<typename TL, typename TR>
	static inline typename std::enable_if<is_indexable<TR>::value, void>::type eval(TL & l, TR const &r)
	{
		l[M - 1] = r[M - 1];
		_assign<M - 1>::eval(l, r);
	}
	template<typename TFun, typename TL, typename TR>
	static inline typename std::enable_if<is_indexable<TR>::value, void>::type eval(TFun const & fun, TL & l,
	        TR const &r)
	{
		l[M - 1] = fun(l[M - 1], r[M - 1]);
		_assign<M - 1>::eval(fun, l, r);
	}
	template<typename TL, typename TR>
	static inline typename std::enable_if<!is_indexable<TR>::value, void>::type eval(TL & l, TR const &r)
	{
		l[M - 1] = r;
		_assign<M - 1>::eval(l, r);
	}
	template<typename TFun, typename TL, typename TR>
	static inline typename std::enable_if<!is_indexable<TR>::value, void>::type eval(TFun const & fun, TL & l,
	        TR const &r)
	{
		l[M - 1] = fun(l[M - 1], r);
		_assign<M - 1>::eval(fun, l, r);
	}
};
template<> struct _assign<1>
{
	template<typename TL, typename TR>
	static inline typename std::enable_if<is_indexable<TR>::value, void>::type eval(TL & l, TR const &r)
	{
		l[0] = r[0];
	}
	template<typename TL, typename TR>
	static inline typename std::enable_if<!is_indexable<TR>::value, void>::type eval(TL & l, TR const &r)
	{
		l[0] = r;
	}
	template<typename TFun, typename TL, typename TR>

	static inline typename std::enable_if<is_indexable<TR>::value, void>::type eval(TFun const & fun, TL & l,
	        TR const &r)
	{
		l[0] = fun(l[0], r[0]);
	}

	template<typename TFun, typename TL, typename TR>
	static inline typename std::enable_if<!is_indexable<TR>::value, void>::type eval(TFun const & fun, TL & l,
	        TR const &r)
	{
		l[0] = fun(l[0], r);
	}
};

template<int N, typename TL, typename TR>
void assign(TL & l, TR const & r)
{
	_assign<N>::eval(l, r);
}
template<int N, typename TFun, typename TL, typename TR>
void assign(TFun const & fun, TL & l, TR const & r)
{
	_assign<N>::eval(fun, l, r);
}
template<int N>
struct _inner_product
{
	template<typename TPlus, typename TMultiplies, typename TL, typename TR>
	static inline auto eval(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l, TR const &r)
	DECL_RET_TYPE((plus_op(multi_op(l[N-1],r[N-1]),_inner_product<N-1>::eval(plus_op,multi_op,l,r))))

};

template<>
struct _inner_product<1>
{
	template<typename TPlus, typename TMultiplies, typename TL, typename TR>
	static inline auto eval(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l, TR const &r)
	DECL_RET_TYPE((multi_op(l[0],r[0])))

};
template<int N, typename TPlus, typename TMultiplies, typename TL, typename TR>
auto inner_product(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l, TR const & r)
DECL_RET_TYPE(_inner_product<N>::eval(plus_op,multi_op,l, r))

}
// namespace ntuple_impl

//--------------------------------------------------------------------------------------------
template<int N, typename T>
struct nTuple
{
	static const int NDIMS = N;
	typedef nTuple<NDIMS, T> this_type;
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
		return _ntuple_impl::inner_product<N>(ops::logical_and(), ops::equal_to(), *this, rhs);
	}

	template<typename TExpr>
	inline bool operator !=(nTuple<NDIMS, TExpr> const &rhs) const
	{
		return _ntuple_impl::inner_product<N>(ops::logical_and(), ops::not_equal_to(), *this, rhs);
	}
	template<typename TR>
	inline operator nTuple<N,TR>() const
	{
		nTuple<N, TR> res;

		_ntuple_impl::assign<N>(_ntuple_impl::binary_right(), res, data_);

		return (res);
	}

	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		_ntuple_impl::assign<N>(_ntuple_impl::binary_right(), data_, rhs);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR rhs[])
	{
		_ntuple_impl::assign<N>(_ntuple_impl::binary_right(), data_, rhs);

		return (*this);
	}
	template<typename TR> inline this_type &
	operator =(nTuple<N, TR> const &rhs)
	{
		_ntuple_impl::assign<N>(_ntuple_impl::binary_right(), data_, rhs);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &rhs)
	{

		_ntuple_impl::assign<N>(ops::plus(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(ops::minus(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(ops::multiplies(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(ops::divides(), data_, rhs);
		return (*this);
	}

	template<int NR, typename TR>
	void operator*(nTuple<NR, TR> const & rhs) = delete;

	template<int NR, typename TR>
	void operator/(nTuple<NR, TR> const & rhs) = delete;

};
template<typename TV>
struct is_nTuple
{
	static constexpr bool value = false;

};

template<int N, typename TV>
struct is_nTuple<nTuple<N, TV>>
{
	static constexpr bool value = true;

};

template<typename TV>
struct nTupleTraits
{
	static constexpr unsigned int NDIMS = 1;
	typedef TV value_type;
};

template<int N, typename TV>
struct nTupleTraits<nTuple<N, TV>>
{
	static constexpr unsigned int NDIMS = N;
	typedef TV value_type;

};

template<int N, class T>
class is_indexable<nTuple<N, T> >
{
public:
	static const bool value = true;

};
//***********************************************************************************
template<int N, typename TL, typename TR>
inline auto Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE ((_ntuple_impl::inner_product<N>(ops::plus(), ops::multiplies(), l, r)))

template<typename T> inline auto Determinant(nTuple<3, nTuple<3, T> > const & m)
DECL_RET_TYPE(( m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
		* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
		* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
)

)

template<typename T> inline auto Determinant(nTuple<4, nTuple<4, T> > const & m) DECL_RET_TYPE(
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

template<int N, typename T> auto abs(nTuple<N, T> const & m)
DECL_RET_TYPE((std::sqrt(std::abs(Dot(m, m)))))

template<int N, int M, typename T> Real abs(nTuple<N, nTuple<M, T>> const & m)
{
	T res = 0.0;
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < M; ++j)
		{
			res += m[i][j] * m[i][j];
		}

	return (std::sqrt(abs(res)));
}

template<typename T> inline
auto real(nTuple<3, T> const & l)
->typename std::enable_if<is_complex<T>::value,nTuple<3,decltype(std::real(l[0]))>>::type
{
	nTuple<3, decltype(std::real(l[0]))> res = { std::real(l[0]), std::real(l[1]), std::real(l[2]) };
	return std::move(res);
}

template<typename T> inline
auto imag(nTuple<3, T> const & l)
->typename std::enable_if<is_complex<T>::value,nTuple<3,decltype(std::real(l[0]))>>::type
{
	nTuple<3, decltype(std::real(l[0]))> res = { std::imag(l[0]), std::imag(l[1]), std::imag(l[2]) };
	return std::move(res);

}

template<typename T> inline
auto real(nTuple<3, T> const & l)
->typename std::enable_if<!is_complex<T>::value,nTuple<3,T> const &>::type
{
	return l;
}

template<typename T> inline
auto imag(nTuple<3, T> const & l)
->typename std::enable_if<!is_complex<T>::value,nTuple<3,T> const &>::type
{
	nTuple<3, T> res = { 0, 0, 0 };
	return l;
}

}
//namespace simpla
#endif  // INCLUDE_NTUPLE_H_
