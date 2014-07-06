/*
 * ntuple_noet.h
 *
 *  Created on: 2014-4-1
 *      Author: salmon
 */

#ifndef NTUPLE_NOET_H_
#define NTUPLE_NOET_H_

namespace simpla
{

template<int N, typename T> struct nTuple;

template<int N, typename T> using Matrix=nTuple<N,nTuple<N,T>>;

template<int N, typename TL>
auto operator -(nTuple<N, TL> const & lhs)
->nTuple<N ,typename std::remove_cv<typename std::remove_reference<decltype(lhs[0])>::type>::type>
{
	typedef typename std::remove_cv<typename std::remove_reference<decltype(lhs[0])>::type>::type T;

	nTuple<N, T> res;

	for (int i = 0; i < N; ++i)
	{
		res[i] = -lhs[i];
	}
	return std::move(res);
}

template<int N, typename TL>
nTuple<N, TL> operator +(nTuple<N, TL> const & lhs)
{
	return std::move(lhs);
}

#define DEFINE_OP(_OP_)                                                          \
template<int N, typename TL, typename TR>                                        \
auto operator _OP_ (nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)        \
->nTuple<N ,decltype(lhs[0] _OP_ rhs[0])>                                        \
{                                                                                \
	nTuple<N, decltype(lhs[0] _OP_ rhs[0])> res;                                 \
                                                                                 \
	for (int i = 0; i < N; ++i)                                                  \
	{                                                                            \
		res[i] = lhs[i] _OP_ rhs[i];                                             \
	}                                                                            \
	return std::move(res);                                                       \
}                                                                                \
template<int N, typename TL, typename TR>                                        \
auto operator _OP_ (nTuple<N, TL> const & lhs, TR const & rhs)        \
->nTuple<N ,decltype(lhs[0] _OP_ rhs )>                                        \
{                                                                                \
	nTuple<N, decltype(lhs[0] _OP_ rhs )> res;                                 \
                                                                                 \
	for (int i = 0; i < N; ++i)                                                  \
	{                                                                            \
		res[i] = lhs[i] _OP_ rhs ;                                             \
	}                                                                            \
	return std::move(res);                                                       \
}                                                                                \
template<int N, typename TL, typename TR>                                        \
auto operator _OP_ (TL const & lhs, nTuple<N, TR> const & rhs)        \
->nTuple<N ,decltype(lhs _OP_ rhs[0])>                                        \
{                                                                                \
	nTuple<N, decltype(lhs  _OP_ rhs[0])> res;                                 \
                                                                                 \
	for (int i = 0; i < N; ++i)                                                  \
	{                                                                            \
		res[i] = lhs  _OP_ rhs[i];                                             \
	}                                                                            \
	return std::move(res);                                                       \
}

DEFINE_OP(+)
DEFINE_OP(-)
DEFINE_OP(*)
DEFINE_OP(/)
DEFINE_OP(&)
DEFINE_OP(|)
#undef DEFINE_OP

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
}
  // namespace simpla

#endif /* NTUPLE_NOET_H_ */
