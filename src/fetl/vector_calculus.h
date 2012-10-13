/*
 *  _fetl_impl::vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_
#include "fetl_defs.h"
#include "typeconvert.h"

namespace simpla
{

namespace _impl
{
template<typename TL, typename TR> struct OpDot
{
	typedef typename simpla::_impl::TypeConvertTraits<
			typename simpla::_impl::nTupleValueTraits<
					typename simpla::_impl::ValueTraits<TL>::ValueType>::ValueType,
			typename simpla::_impl::nTupleValueTraits<
					typename simpla::_impl::ValueTraits<TR>::ValueType>::ValueType>::ValueType ValueType;
};

template<typename TL, typename TR> struct OpCross
{
	typedef nTuple<THREE,
			typename simpla::_impl::TypeConvertTraits<
					typename simpla::_impl::nTupleValueTraits<
							typename simpla::_impl::ValueTraits<TL>::ValueType>::ValueType,
					typename simpla::_impl::nTupleValueTraits<
							typename simpla::_impl::ValueTraits<TR>::ValueType>::ValueType>::ValueType> ValueType;
	;
};

template<typename TL> struct OpGrad
{
	typedef typename TL::ValueType ValueType;
};
template<typename TL> struct OpDiverge
{
	typedef typename TL::ValueType ValueType;
};
template<typename TL> struct OpCurl
{
	typedef typename TL::ValueType ValueType;
};
template<typename IR, typename TL> struct OpCurlPD
{
	typedef typename TL::ValueType ValueType;
};

} // namespace namespace _impl{

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpDot<Field<TG, IZeroForm, TLExpr>, Field<TG, IZeroForm, TRExpr> > >  //
Dot(Field<TG, IZeroForm, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{
	return (Field<TG, IZeroForm,
			_impl::OpDot<Field<TG, IZeroForm, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, int N, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpDot<nTuple<N, TLExpr>, Field<TG, IZeroForm, TRExpr> > >      //
Dot(nTuple<N, TLExpr> const & lhs, Field<TG, IZeroForm, TRExpr> const &rhs)
{

	return (Field<TG, IZeroForm,
			_impl::OpDot<nTuple<N, TLExpr>, Field<TG, IZeroForm, TRExpr> > >(
			lhs, rhs));
}

template<typename TG, int N, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpDot<Field<TG, IZeroForm, TLExpr>, nTuple<N, TRExpr> > >      //
Dot(Field<TG, IZeroForm, TLExpr> const & lhs, nTuple<N, TRExpr> const & rhs)
{

	return (Field<TG, IZeroForm,
			_impl::OpDot<Field<TG, IZeroForm, TLExpr>, nTuple<N, TRExpr> > >(
			lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpCross<Field<TG, IZeroForm, TLExpr>,
				Field<TG, IZeroForm, TRExpr> > >                              //
Cross(Field<TG, IZeroForm, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const & rhs)
{
	return (Field<TG, IZeroForm,
			_impl::OpCross<Field<TG, IZeroForm, TLExpr>,
					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpCross<nTuple<THREE, TLExpr>, Field<TG, IZeroForm, TRExpr> > >  //
Cross(nTuple<THREE, TLExpr> const & lhs,
		Field<TG, IZeroForm, TRExpr> const &rhs)
{
	return (Field<TG, IZeroForm,
			_impl::OpCross<nTuple<THREE, TLExpr>, Field<TG, IZeroForm, TRExpr> > >(
			lhs, rhs));
}

template<typename TG, typename TLExpr, typename TRExpr>
inline Field<TG, IZeroForm,
		_impl::OpCross<Field<TG, IZeroForm, TLExpr>, nTuple<THREE, TRExpr> > >  //
Cross(Field<TG, IZeroForm, TLExpr> const & lhs,
		nTuple<THREE, TRExpr> const & rhs)
{
	return (Field<TG, IZeroForm,
			_impl::OpCross<Field<TG, IZeroForm, TLExpr>, nTuple<THREE, TRExpr> > >(
			lhs, rhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, IOneForm, _impl::OpGrad<Field<TG, IZeroForm, TLExpr> > >     //
Grad(Field<TG, IZeroForm, TLExpr> const & lhs)
{
	return (Field<TG, IOneForm, _impl::OpGrad<Field<TG, IZeroForm, TLExpr> > >(
			lhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, IZeroForm, _impl::OpDiverge<Field<TG, IOneForm, TLExpr> > >  //
Diverge(Field<TG, IOneForm, TLExpr> const & lhs)
{
	return (Field<TG, IZeroForm, _impl::OpDiverge<Field<TG, IOneForm, TLExpr> > >(
			lhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, ITwoForm, _impl::OpCurl<Field<TG, IOneForm, TLExpr> > >      //
Curl(Field<TG, IOneForm, TLExpr> const & lhs)
{
	return (Field<TG, ITwoForm, _impl::OpCurl<Field<TG, IOneForm, TLExpr> > >(
			lhs));
}

template<typename TG, typename TLExpr>
inline Field<TG, IOneForm, _impl::OpCurl<Field<TG, ITwoForm, TLExpr> > >      //
Curl(Field<TG, ITwoForm, TLExpr> const & lhs)
{
	return (Field<TG, IOneForm, _impl::OpCurl<Field<TG, ITwoForm, TLExpr> > >(
			lhs));
}

template<int IPD, typename TG, typename TLExpr>
inline Field<TG, ITwoForm,
		_impl::OpCurlPD<Int2Type<IPD>, Field<TG, IOneForm, TLExpr> > >        //
CurlPD(Int2Type<IPD>, Field<TG, IOneForm, TLExpr> const & lhs)
{
	return (Field<TG, ITwoForm,
			_impl::OpCurlPD<Int2Type<IPD>, Field<TG, IOneForm, TLExpr> > >(
			Int2Type<IPD>(), lhs));
}

template<int IPD, typename TG, typename TLExpr>
inline Field<TG, IOneForm,
		_impl::OpCurlPD<Int2Type<IPD>, Field<TG, ITwoForm, TLExpr> > >        //
CurlPD(Int2Type<IPD>, Field<TG, ITwoForm, TLExpr> const & lhs)
{
	return (Field<TG, IOneForm,
			_impl::OpCurlPD<Int2Type<IPD>, Field<TG, ITwoForm, TLExpr> > >(
			Int2Type<IPD>(), lhs));
}

}        // namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
