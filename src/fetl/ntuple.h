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
//#include <sstream>
//#include <string>

#include "primitives.h"

//#include <type_traits>

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
	static const int NUM_OF_DIMS = N;
	typedef nTuple<NUM_OF_DIMS, T> this_type;
	typedef typename std::remove_reference<T>::type value_type;

	value_type v_[N];

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
		return (v_[i]);
	}

	inline value_type const&
	operator[](size_t i) const
	{
		return (v_[i]);
	}

	template<typename TR>
	inline operator nTuple<N,TR>() const
	{
		nTuple<N, TR> res;
		ntuple_impl::_assign<N, nTuple<N, TR>, this_type>::eval(res, v_);
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
	inline bool operator !=(nTuple<NUM_OF_DIMS, TExpr> const &rhs) const
	{
		return (!(*this == rhs));
	}
//	template<typename TR> inline typename std::enable_if<
//	!is_indexable<TR>::value, ThisType &>::type //
//	operator =(TR const &rhs)
//	{
//		for (int i = 0; i < NDIM; ++i)
//		{
//			v_[i] = rhs;
//		}
//		return (*this);
//	}

	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		for (int i = 0; i < N; ++i)
		{
			v_[i] = rhs;
		}

		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR rhs[])
	{
		for (int i = 0; i < N; ++i)
		{
			v_[i] = rhs[i];
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
};

template<typename T> struct is_nTuple
{
	static const bool value = false;
};

template<int N, typename T> struct is_nTuple<nTuple<N, T> >
{
	static const bool value = true;
};

template<int N, typename T>
struct is_storage_type<nTuple<N, T> >
{
	static const bool value = is_storage_type<T>::value;
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

//template<int N>
//struct is_storage_type<nTuple<N, double> >
//{
//	static const bool value = true;
//};
//
//template<int N>
//struct is_storage_type<nTuple<N, int> >
//{
//	static const bool value = true;
//};
//template<int N>
//struct is_storage_type<nTuple<N, std::complex<double> > >
//{
//	static const bool value = true;
//};

}
//namespace simpla
#endif  // INCLUDE_NTUPLE_H_
