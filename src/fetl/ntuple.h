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

namespace ntuple_impl
{

template<int M, typename TL, typename TR> struct _swap
{
	static inline void eval(TL const & l, TR const &r)
	{
		std::swap(l[M - 1], r[M - 1]);
		_swap<M - 1, TL, TR>::eval(l, r);
	}
};
template<typename TL, typename TR> struct _swap<1, TL, TR>
{
	static inline void eval(TL const & l, TR const &r)
	{
		std::swap(l[0], r[0]);
	}
};

template<int M, typename TL, typename TR> struct _assign
{
	static inline void eval(TL & l, TR const &r)
	{
		l[M - 1] = r[M - 1];
		_assign<M - 1, TL, TR>::eval(l, r);
	}
};
template<typename TL, typename TR> struct _assign<1, TL, TR>
{
	static inline void eval(TL & l, TR const &r)
	{
		l[0] = r[0];
	}
};

template<int M, typename TL, typename TR> struct _equal
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE((l[M - 1] ==r[M - 1] && _equal<M - 1, TL, TR>::eval(l, r)))
};
template<typename TL, typename TR> struct _equal<1, TL, TR>
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE(l[0]==r[0])
};
}  // namespace _impl
//--------------------------------------------------------------------------------------------
template<int N, typename T>
struct nTuple
{
	static const int NDIMS = N;
	typedef nTuple<NDIMS, T> this_type;
	typedef T value_type;

	value_type data_[N];

//	nTuple()
//	{
//	}
//	nTuple(std::initializer_list<T> r)
//	{
//		int i = 0;
//		auto it = r.begin();
//		for (; i < N && it != r.end(); ++it, ++i)
//		{
//			v_[i] = *it;
//		}
//	}
//

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

	template<typename TR>
	inline operator nTuple<N,TR>() const
	{
		nTuple<N, TR> res;
		ntuple_impl::_assign<N, nTuple<N, TR>, this_type>::eval(res, data_);
		return (res);
	}

	inline void swap(this_type & rhs)
	{
		ntuple_impl::_swap<N, this_type, this_type>::eval(*this, rhs);
	}

	template<typename TR>
	inline bool operator ==(TR const &rhs) const
	{
		return (ntuple_impl::_equal<N, this_type, TR>::eval(*this, rhs));
	}

	template<typename TExpr>
	inline bool operator !=(nTuple<NDIMS, TExpr> const &rhs) const
	{
		return (!(*this == rhs));
	}

	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		for (int i = 0; i < N; ++i)
		{
			data_[i] = rhs;
		}

		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR rhs[])
	{
		for (int i = 0; i < N; ++i)
		{
			data_[i] = rhs[i];
		}

		return (*this);
	}
	template<typename TR> inline this_type &
	operator =(nTuple<N, TR> const &rhs)
	{
		ntuple_impl::_assign<N, this_type, nTuple<N, TR>>::eval(*this, rhs);

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &rhs)
	{
		*this = *this + rhs;
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		*this = *this - rhs;
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		*this = (*this) * rhs;
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		*this = *this / rhs;
		return (*this);
	}

	template<typename TR>
	auto operator+(nTuple<NDIMS, TR> const & rhs)
	->nTuple<NDIMS,decltype(std::declval<value_type>()+ std::declval<TR>())>
	{
		nTuple<NDIMS, decltype(std::declval<value_type>()+ std::declval<TR>())> res;
		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = data_[i] + rhs[i];
		}
		return std::move(res);
	}

	template<typename TR>
	auto operator-(nTuple<NDIMS, TR> const & rhs)
	->nTuple<NDIMS,decltype(std::declval<value_type>()- std::declval<TR>())>
	{
		nTuple<NDIMS, decltype(std::declval<value_type>()- std::declval<TR>())> res;
		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = data_[i] - rhs[i];
		}
		return std::move(res);
	}

	auto operator-()->nTuple<NDIMS,decltype(-std::declval<value_type>() )>
	{
		nTuple<NDIMS, decltype(- std::declval<value_type>() )> res;
		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = -data_[i];
		}
		return std::move(res);
	}

	this_type const & operator+()
	{
		return *this;
	}
	template<int NR, typename TR>
	void operator*(nTuple<NR, TR> const & rhs) = delete;

	template<int NR, typename TR>
	void operator/(nTuple<NR, TR> const & rhs) = delete;

	template<typename TR>
	auto operator*(TR const & rhs)
	->nTuple<NDIMS,decltype(std::declval<value_type>()* std::declval<TR>())>
	{
		nTuple<NDIMS, decltype(std::declval<value_type>()* std::declval<TR>())> res;
		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = data_[i] * rhs;
		}
		return std::move(res);
	}

	template<typename TR>
	auto operator/(TR const & rhs)
	->nTuple<NDIMS,decltype(std::declval<value_type>()/ std::declval<TR>())>
	{
		nTuple<NDIMS, decltype(std::declval<value_type>()/ std::declval<TR>())> res;
		for (int i = 0; i < NDIMS; ++i)
		{
			res[i] = data_[i] / rhs;
		}
		return std::move(res);
	}
};


template<typename T> struct is_nTuple
{
	static const bool value = false;
};

template<int N, typename T> struct is_nTuple<nTuple<N, T> >
{
	static const bool value = true;
};

template<typename T>
struct nTupleTraits
{
	static const int NUM_OF_DIMS = 1;
	typedef T value_type;
};
template<int N, typename T>
struct nTupleTraits<nTuple<N, T>>
{
	static const int NUM_OF_DIMS = N;
	typedef T value_type;
};

template<int N, typename TL, typename TR>
inline auto InnerProduct(nTuple<N, TL> const &l, nTuple<N, TR> const &r)->decltype(l[0]*r[0])
{

	decltype(l[0]*r[0]) res = 0;

	for (int i = 0; i < N; ++i)
		res += l[i] * r[i];

	return std::move(res);

}

template<int N, typename TL, typename TR>
inline auto Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((InnerProduct(l,r)))

//***********************************************************************************

template<typename TL, typename TR> inline auto Cross(nTuple<3, TL> const & l, nTuple<3, TR> const & r)
->nTuple<3,decltype(l[0] * r[0])>
{
	nTuple<3, decltype(l[0] * r[0])> res;
	res[0] = l[1] * r[2] - l[2] * r[1];
	res[1] = l[2] * r[0] - l[0] * r[2];
	res[2] = l[0] * r[1] - l[1] * r[0];
	return std::move(res);
}

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

template<int N, typename T> auto abs(simpla::nTuple<N, T> const & m)
DECL_RET_TYPE((std::sqrt(std::abs(Dot(m, m)))))

//namespace ntuple_impl
//{
//
//template<int N, typename T, typename TI>
//inline auto OpEval(Int2Type<REAL>, nTuple<N, T> const & l, TI const & s)
//DECL_RET_TYPE ((std::real(l[s]) ))
//
//template<int N, typename T, typename TI>
//inline auto OpEval(Int2Type<IMAGINE>, nTuple<N, T> const & l, TI const & s)
//DECL_RET_TYPE ((std::imag(l[s]) ))
//
//}  // namespace ntuple_impl
//
//template<int N, typename TL> inline
//auto real(nTuple<N, TL> const & l)
//DECL_RET_TYPE(( nTuple<N, UniOp<REAL,nTuple<N, TL> > > (l)))
//
//template<int N, typename TL> inline
//auto imag(nTuple<N, TL> const & l)
//DECL_RET_TYPE(( nTuple<N, UniOp<IMAGINE,nTuple<N, TL> > > (l)))

//@TODO add real and imag for generic nTuple

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
