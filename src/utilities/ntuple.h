/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * nTuple.h
 *
 *  Created on: Jan 27, 2010
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
template<unsigned int N, typename TL, typename TR>
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
template<unsigned int M> struct _assign
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
struct _inner_product
{
	template<typename TPlus, typename TMultiplies, typename TL, typename TR>
	static inline auto eval(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l, TR const &r)
	DECL_RET_TYPE ((plus_op(multi_op(l[N - 1], r[N - 1]), _inner_product<N - 1>::eval(plus_op, multi_op, l, r))))

};

template<>
struct _inner_product<1>
{
	template<typename TPlus, typename TMultiplies, typename TL, typename TR>
	static inline auto eval(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l, TR const &r)
	DECL_RET_TYPE ((multi_op(l[0], r[0])))

};
template<unsigned int N, typename TPlus, typename TMultiplies, typename TL, typename TR>
auto inner_product(TPlus const & plus_op, TMultiplies const &multi_op, TL const & l, TR const & r)
DECL_RET_TYPE (_inner_product<N>::eval(plus_op, multi_op, l, r))

}
// namespace ntuple_impl
/**
 * \ingroup DataStruct
 * \brief nTuple :n-tuple
 *
 *   Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a positive
 *      unsigned int   eger. There is also one 0-tuple, an empty sequence. An n-tuple is defined
 *    inductively using the construction of an ordered pair. Tuples are usually
 *    written by listing the elements within parenthesis '( )' and separated by
 *    commas; for example, (2, 7, 4, 1, 7) denotes a 5-tuple.
 *    \cite  http://en.wikipedia.org/wiki/Tuple
 *   Implement
 *   template< unsigned int     n,typename T> struct nTuple;
 *   nTuple<5,double> t={1,2,3,4,5};
 *
 **/
template<unsigned int N, typename T>
struct nTuple
{
	static const unsigned int NDIMS = N;
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
		return _ntuple_impl::inner_product<N>(std::logical_and<void>(), std::equal_to<T>(), *this, rhs);
	}

	template<typename TExpr>
	inline bool operator !=(nTuple<NDIMS, TExpr> const &rhs) const
	{
		return _ntuple_impl::inner_product<N>(std::logical_and<void>(), std::not_equal_to<T>(), *this, rhs);
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

		_ntuple_impl::assign<N>(std::plus<void>(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(std::minus<void>(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(std::multiplies<void>(), data_, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		_ntuple_impl::assign<N>(std::divides<void>(), data_, rhs);
		return (*this);
	}

	template<unsigned int NR, typename TR>
	void operator*(nTuple<NR, TR> const & rhs) = delete;

	template<unsigned int NR, typename TR>
	void operator/(nTuple<NR, TR> const & rhs) = delete;

};

template<typename T>
auto make_ntuple(T v0)
DECL_RET_TYPE(v0)
;

template<typename T>
nTuple<2, T> make_ntuple(T v0, T v1)
{
	return std::move(nTuple<2, T>( { v0, v1 }));
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

template<typename TV>
struct is_nTuple
{
	static constexpr bool value = false;

};

template<unsigned int N, typename TV>
struct is_nTuple<nTuple<N, TV>>
{
	static constexpr bool value = true;

};

template<typename T>
struct is_ntuple_expression
{
	static constexpr bool value = false;
};

template<unsigned int N, unsigned int TOP, typename TL, typename TR>
struct is_ntuple_expression<nTuple<N, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<unsigned int N, unsigned int TOP, typename TL, typename TR>
struct is_expression<nTuple<N, BiOp<TOP, TL, TR> > >
{
	static constexpr bool value = true;
};

template<unsigned int N, typename TE>
struct is_primitive<nTuple<N, TE> >
{
	static constexpr bool value = is_arithmetic_scalar<TE>::value;
};

template<typename TV>
struct nTupleTraits
{
	static constexpr unsigned int NDIMS = 1;
	typedef TV value_type;
	typedef value_type element_type;
	template<typename TVec>
	static void get_dimensions(TVec* dims)
	{
	}

};

template<unsigned int N, typename TV>
struct nTupleTraits<nTuple<N, TV>>
{
	static constexpr unsigned int NDIMS = N;

	typedef TV value_type;

	typedef typename nTupleTraits<TV>::element_type element_type;

	template<typename TVec>
	static void get_dimensions(TVec* dims)
	{
		dims->push_back(NDIMS);
		nTupleTraits<TV>::get_dimensions(dims);
	}
};

template<unsigned int N, class T>
class is_indexable<nTuple<N, T> >
{
public:
	static const bool value = true;

};
//***********************************************************************************
template<unsigned int N, typename TL, typename TR>
inline auto Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE ((_ntuple_impl::inner_product<N>(std::plus<void>(), std::multiplies<void>(), l, r)))

template<unsigned int N, typename TL, typename TR>
inline auto InnerProductNTuple(nTuple<N, TL> const &l, nTuple<N, TR> const &r)->decltype(l[0]*r[0])
{
	decltype(l[0]*r[0]) res = 0;
	for (unsigned int i = 0; i < N; ++i)
		res += l[i] * r[i];
	return res;
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

template<unsigned int N, typename T> auto abs(nTuple<N, T> const & m)
DECL_RET_TYPE((std::sqrt(std::abs(Dot(m, m)))))

template<unsigned int N, typename T> inline auto NProduct(nTuple<N, T> const & v)
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
template<unsigned int N, unsigned int M, typename T> Real abs(nTuple<N, nTuple<M, T>> const & m)
{
	T res = 0.0;
	for (unsigned int i = 0; i < N; ++i)
		for (unsigned int j = 0; j < M; ++j)
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

template<typename T> inline constexpr nTuple<3, T> real(nTuple<3, std::complex<T>> const &v)
{
	return std::move(nTuple<3, T>( { v[0].real(), v[1].real(), v[2].real() }));
}

template<typename T> inline constexpr nTuple<3, T> imag(nTuple<3, std::complex<T>> const &v)
{
	return std::move(nTuple<3, T>( { v[0].imag(), v[1].imag(), v[2].imag() }));
}

template<unsigned int NDIMS, typename TExpr>
auto operator >>(nTuple<NDIMS, TExpr> const & v, unsigned int n)-> nTuple<NDIMS,decltype(v[0] >> n )>
{
	nTuple<NDIMS, decltype(v[0] >> n )> res;
	for (unsigned int i = 0; i < NDIMS; ++i)
	{
		res[i] = v[i] >> n;
	}
	return res;
}

template<unsigned int NDIMS, typename TExpr>
auto operator <<(nTuple<NDIMS, TExpr> const & v, unsigned int n)-> nTuple<NDIMS,decltype(v[0] << n )>
{
	nTuple<NDIMS, decltype(v[0] >> n )> res;
	for (unsigned int i = 0; i < NDIMS; ++i)
	{
		res[i] = v[i] << n;
	}
	return res;
}

template<unsigned int N, typename T> inline nTuple<N, T> real(nTuple<N, std::complex<T>> const &v)
{
	nTuple<N, T> res;
	for (unsigned int i = 0; i < N; ++i)
	{
		res[i] = v[i].real();
	}
	return std::move(res);
}

template<unsigned int N, typename T> inline nTuple<N, T> imag(nTuple<N, std::complex<T>> const &v)
{
	nTuple<N, T> res;
	for (unsigned int i = 0; i < N; ++i)
	{
		res[i] = v[i].imag();
	}
	return std::move(res);
}

}
//namespace simpla

//namespace std
//{
//template<typename > struct hash;
//template<>
//template< unsigned int     N,typename TV>
//struct hash<simpla::nTuple<N, TV>>
//{
//	typedef S argument_type;
//	typedef std::size_t value_type;
//
//	value_type operator()(argument_type const& s) const
//	{
//		value_type const h1(std::hash<std::string>()(s.first_name));
//		value_type const h2(std::hash<std::string>()(s.last_name));
//		return h1 ^ (h2 << 1);
//	}
//}
//}  // namespace std
#endif  // INCLUDE_NTUPLE_H_
