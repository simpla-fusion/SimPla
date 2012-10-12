/*
 *  _fetl_impl::vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_
#include "fetl_defs.h"
#include "ntuple.h"

namespace simpla
{
namespace fetl
{

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpDot<Field<TG, IZeroForm, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >                              //
Dot(Field<TG, IZeroForm, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{
	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpDot<Field<TG, IZeroForm, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, int N, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpDot<nTuple<N, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >          //
Dot(nTuple<N, TLExpr> const & lhs, Field<TG, IZeroForm, TRExpr> const &rhs)
{

	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpDot<nTuple<N, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, int N, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpDot<Field<TG, IZeroForm, TLExpr>,
				nTuple<N, TRExpr> > >                                         //
Dot(Field<TG, IZeroForm, TLExpr> const & lhs, nTuple<N, TRExpr> const & rhs)
{

	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpDot<Field<TG, IZeroForm, TLExpr>,
					nTuple<N, TRExpr> > >(lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpCross<Field<TG, IZeroForm, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >                              //
Cross(Field<TG, IZeroForm, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{
	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpCross<Field<TG, IZeroForm, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpCross<nTuple<THREE, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >                              //
Cross(nTuple<THREE, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const &rhs)
{
	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpCross<nTuple<THREE, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpCross<Field<TG, IZeroForm, TLExpr>,
				nTuple<THREE, TRExpr> > >                                     //
Cross(Field<TG, IZeroForm, TLExpr> const & lhs,
		nTuple<THREE, TRExpr> const & rhs)
{
	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpCross<Field<TG, IZeroForm, TLExpr>,
					nTuple<THREE, TRExpr> > >(lhs, rhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, IOneForm,
		_fetl_impl::vector_calculus::OpGrad<Field<TG, IZeroForm, TLExpr> > >  //
Grad(Field<TG, IZeroForm, TLExpr> const & lhs)
{
	return (Field<TG, IOneForm,
			_fetl_impl::vector_calculus::OpGrad<Field<TG, IZeroForm, TLExpr> > >(
			lhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, IZeroForm,
		_fetl_impl::vector_calculus::OpDiverge<Field<TG, IOneForm, TLExpr> > >  //
Diverge(Field<TG, IOneForm, TLExpr> const & lhs)
{
	return (Field<TG, IZeroForm,
			_fetl_impl::vector_calculus::OpDiverge<Field<TG, IOneForm, TLExpr> > >(
			lhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, ITwoForm,
		_fetl_impl::vector_calculus::OpCurl<Field<TG, IOneForm, TLExpr> > >   //
Curl(Field<TG, IOneForm, TLExpr> const & lhs)
{
	return (Field<TG, ITwoForm,
			_fetl_impl::vector_calculus::OpCurl<Field<TG, IOneForm, TLExpr> > >(
			lhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, IOneForm,
		_fetl_impl::vector_calculus::OpCurl<Field<TG, ITwoForm, TLExpr> > >   //
Curl(Field<TG, ITwoForm, TLExpr> const & lhs)
{
	return (Field<TG, IOneForm,
			_fetl_impl::vector_calculus::OpCurl<Field<TG, ITwoForm, TLExpr> > >(
			lhs));
}

template<typename TG, int IFORM_OUT, int IFORM_IN, int IPD, typename TLExpr>
struct Field<TG, IFORM_OUT,
		_fetl_impl::vector_calculus::OpCurlPD<IPD, Field<TG, IFORM_IN, TLExpr> > >
{
	typedef Field<TG, IFORM_IN, TLExpr> TL;

	typename TypeTraits<TL>::ConstReference lhs_;

	static const int IForm = ITwoForm;

	typedef typename TL::ValueType ValueType;

	typedef Field<TG, IFORM_OUT, _fetl_impl::vector_calculus::OpCurlPD<IPD, TL> > ThisType;

	typedef typename TL::Grid Grid;

	Grid const & grid;

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (lhs_.grid.curlPd_<IPD>(lhs_, s));
	}
};
template<typename TG, int IPD, typename TLExpr>
inline Field<TG, ITwoForm,
		_fetl_impl::vector_calculus::OpCurlPD<IPD, Field<TG, IOneForm, TLExpr> > >  //
CurlPD(Field<TG, IOneForm, TLExpr> const & lhs)
{
	return (Field<TG, ITwoForm,
			_fetl_impl::vector_calculus::OpCurlPD<IPD,
					Field<TG, IOneForm, TLExpr> > >(lhs));
}

template<typename TG, int IPD, typename TLExpr>
inline Field<TG, IOneForm,
		_fetl_impl::vector_calculus::OpCurlPD<IPD, Field<TG, ITwoForm, TLExpr> > >  //
CurlPD(Field<TG, ITwoForm, TLExpr> const & lhs)
{
	return (Field<TG, IOneForm,
			_fetl_impl::vector_calculus::OpCurlPD<IPD,
					Field<TG, ITwoForm, TLExpr> > >(lhs));
}

}
// namespace fetl
}// namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
