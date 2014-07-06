/*
 * ntuple_et.h
 *
 *  Created on: 2014-4-1
 *      Author: salmon
 */

#ifndef NTUPLE_ET_H_
#define NTUPLE_ET_H_

#include "primitives.h"

#include <sstream>
#include <string>
#include "complex_ops.h"
namespace simpla
{
// Expression template of nTuple
template<unsigned int N, typename T> struct nTuple;
template<unsigned int N, typename T> using Matrix=nTuple<N,nTuple<N,T>>;
//***********************************************************************************

namespace ntuple_impl
{

template<typename TOP, typename TL, typename TR> struct nTupleBiOp;

template<typename TOP, typename TL> struct nTupleUniOp;

}  // namespace ntuple_impl
template<unsigned int N, typename TOP, typename TL, typename TR>
struct nTuple<N, ntuple_impl::nTupleBiOp<TOP, TL, TR>>
{
	typename StorageTraits<TL>::const_reference l_;
	typename StorageTraits<TR>::const_reference r_;

	nTuple(TL const & l, TR const & r)
			: l_(l), r_(r)
	{
	}

	~nTuple()
	{
	}
	template<typename TI>
	constexpr auto operator[](TI s) const
	DECL_RET_TYPE(ops::eval(TOP(),l_,r_,s))

};
template<unsigned int N, typename TOP, typename TL>
struct nTuple<N, ntuple_impl::nTupleUniOp<TOP, TL>>
{
	typename StorageTraits<TL>::const_reference l_;

	template<typename TI>
	constexpr auto operator[](TI s) const
	DECL_RET_TYPE(ops::eval(TOP(),l_, s))
};
//***********************************************************************************

template<unsigned int N, typename TL> inline
auto operator-(nTuple<N, TL> const & l)
DECL_RET_TYPE(( nTuple<N, ntuple_impl::nTupleUniOp<ops::negate,nTuple<N, TL> > > (l)))

template<unsigned int N, typename TL> inline
auto operator+(nTuple<N, TL> const & f)
DECL_RET_TYPE(f)

template<unsigned int N, typename TL, typename TR> inline auto //
operator +(nTuple<N, TL> const & l, nTuple<N, TR> const & r)
DECL_RET_TYPE(((nTuple<N, ntuple_impl::nTupleBiOp<ops::plus ,nTuple<N, TL>, nTuple<N, TR> > >(l, r))))

template<unsigned int N, typename TL, typename TR> inline auto //
operator -(nTuple<N, TL> const & l, nTuple<N, TR> const & r)
DECL_RET_TYPE(((nTuple<N, ntuple_impl::nTupleBiOp<ops::minus ,nTuple<N, TL>, nTuple<N, TR> > >(l, r))))

template<unsigned int N, typename TL, typename TR> inline auto //
operator *(nTuple<N, TL> const & l, nTuple<N, TR> const & r)
DECL_RET_TYPE(((nTuple<N,ntuple_impl::nTupleBiOp<ops::multiplies, nTuple<N, TL>, nTuple<N, TR> > >(l, r))))

template<unsigned int N, typename TL, typename TR> inline auto //
operator /(nTuple<N, TL> const & l, nTuple<N, TR> const & r)
DECL_RET_TYPE(((nTuple<N, ntuple_impl::nTupleBiOp<ops::divides, nTuple<N, TL>, nTuple<N, TR> > >(l, r))))

//*******************************************************************************************************
// nTuple vs other and other vs nTuple
//*******************************************************************************************************
#define DEF_BIOP(_NAME_,_OP_ , _OTHER_)                                                                 \
template<unsigned int N, typename TL> inline auto operator _OP_(nTuple<N, TL> const & l, _OTHER_ const &  r)    \
DECL_RET_TYPE(((nTuple<N, ntuple_impl::nTupleBiOp<_NAME_, nTuple<N, TL>, _OTHER_ > >(l, r))))          \
template<unsigned int N, typename TR> inline auto operator _OP_(_OTHER_  const & l, nTuple<N, TR> const & r)    \
DECL_RET_TYPE(((nTuple<N,ntuple_impl:: nTupleBiOp<_NAME_, _OTHER_, nTuple<N, TR> > >(l, r))))          \

#define DEF_BIOP_BUNDLE(_OTHER_)                                                                        \
DEF_BIOP(ops::plus, +,  _OTHER_)                                                                        \
DEF_BIOP(ops::minus, -,   _OTHER_)                                                                      \
DEF_BIOP(ops::multiplies ,  *, _OTHER_)                                                                 \
DEF_BIOP(ops::divides,/,  _OTHER_)                                                                      \

DEF_BIOP_BUNDLE(int)
DEF_BIOP_BUNDLE(long)
DEF_BIOP_BUNDLE( unsigned int  )
DEF_BIOP_BUNDLE(unsigned long)
DEF_BIOP_BUNDLE(float)
DEF_BIOP_BUNDLE(double)
DEF_BIOP_BUNDLE(std::complex<float>)
DEF_BIOP_BUNDLE(std::complex<double>)

#undef DEF_BIOP_BUNDLE
#undef DEF_BIOP

//***********************************************************************************
namespace ntuple_impl
{

template<unsigned int M, typename TL, typename TR> struct _inner_product_s;

template<typename TL, typename TR>
inline auto _inner_product(TL const & l, TR const &r)
DECL_RET_TYPE((l*r))

template<unsigned int N, typename TL, typename TR>
inline auto _inner_product(nTuple<N, TL> const & l, nTuple<N, TR> const &r)
DECL_RET_TYPE(( _inner_product_s<N, nTuple<N, TL>, nTuple<N, TR> >::eval(l,r)))

template<unsigned int M, typename TL, typename TR>
struct _inner_product_s
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE((_inner_product(l[M - 1] , r[M - 1]) + _inner_product_s<M - 1, TL, TR>::eval(l, r)))
};
template<typename TL, typename TR>
struct _inner_product_s<1, TL, TR>
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE(_inner_product(l[0],r[0]))
}
;

}
//namespace ntuple_impl
template<unsigned int N, typename TL, typename TR>
inline auto Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((ntuple_impl::_inner_product(l,r)))

template<unsigned int N, typename TL, typename TR>
inline auto InnerProduct(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((ntuple_impl::_inner_product(l,r)))
//***********************************************************************************
namespace ntuple_impl
{
template<unsigned int N, typename TL, typename TR>
inline auto OpEval(Int2Type<CROSS>, nTuple<N, TL> const & l, nTuple<N, TR> const &r, size_t s)
DECL_RET_TYPE ((l[(s+1)%3] * r[(s+2)%3] - l[(s+2)%3] * r[(s+1)%3]))
}  // namespace ntuple_impl

template<unsigned int N, typename TL, typename TR> inline auto Cross(nTuple<N, TL> const & l, nTuple<N, TR> const & r)
DECL_RET_TYPE( (nTuple<N,BiOp<CROSS, nTuple<N, TL>,nTuple<N, TR> > > (l, r)))

template<typename TL, typename TR> inline auto Cross(nTuple<3, TL> const & l, nTuple<3, TR> const & r)
->nTuple<3,decltype(l[0]*r[0])>
{
	nTuple<3, decltype(l[0]*r[0])> res = { l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2], l[0] * r[1]
	        - l[1] * r[0] };
	return std::move(res);
}
//DECL_RET_TYPE( (nTuple<N,BiOp<CROSS, nTuple<N, TL>,nTuple<N, TR> > > (l, r)))

//***********************************************************************************
// overloading operators

}// namespace simpla
#endif /* NTUPLE_ET_H_ */
