/*
 * typeconvert.h
 *
 *  Created on: 2012-3-23
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */

#ifndef TYPECONVERT_H_
#define TYPECONVERT_H_
namespace simpla
{

template<int, typename > struct nTuple;
template<typename TG, int IFORM, typename > struct Field;

// NOTE: this is a replacement before C++11 available for CUDA

namespace _impl
{

template<typename T> struct TypeTraits
{
	typedef T Reference;
	typedef const T ConstReference;
};

template<typename, typename > struct TypeConvertTraits;

template<>
struct TypeConvertTraits<int, int>
{
	typedef int ValueType;
};

template<>
struct TypeConvertTraits<long, long>
{
	typedef long ValueType;
};

template<>
struct TypeConvertTraits<double, double>
{
	typedef double ValueType;
};

template<>
struct TypeConvertTraits<std::complex<double>, std::complex<double> >
{
	typedef std::complex<double> ValueType;
};

template<typename TG, int IFORM, typename TLEXPR, typename TR>
struct TypeConvertTraits<Field<TG, IFORM, TLEXPR>, TR>
{
	typedef typename TypeConvertTraits<
			typename Field<TG, IFORM, TLEXPR>::ValueType, TR>::ValueType ValueType;
};
template<typename TL, typename TG, int IFORM, typename TEXPR>
struct TypeConvertTraits<TL, Field<TG, IFORM, TEXPR> >
{
	typedef typename TypeConvertTraits<TL,
			typename Field<TG, IFORM, TEXPR>::ValueType>::ValueType ValueType;
};

template<typename TG, int ILFORM, typename TLEXPR, int IRFORM, typename TREXPR>
struct TypeConvertTraits<Field<TG, ILFORM, TLEXPR>, Field<TG, IRFORM, TREXPR> >
{
	typedef typename TypeConvertTraits<
			typename Field<TG, ILFORM, TLEXPR>::ValueType,
			typename Field<TG, IRFORM, TREXPR>::ValueType>::ValueType ValueType;
};

#define TYPE_OP_RULE(TL,TR,TV) \
template<> struct TypeConvertTraits<TL, TR> { 	typedef TV ValueType; }; \
template<> struct TypeConvertTraits<TR, TL> { 	typedef TV ValueType; };

TYPE_OP_RULE(double, int, double)
TYPE_OP_RULE(double, long, double)
TYPE_OP_RULE(double, unsigned int, double)
TYPE_OP_RULE(double, unsigned long, double)
TYPE_OP_RULE(std::complex<double>, double, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, int, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, unsigned int, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, long, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, unsigned long, std::complex<double>)
#undef TYPE_OP_RULE

////---------------------------------------------------------------------------------------------

template<typename T> struct ComplexTraits
{
	typedef std::complex<T> ValueType;
};

template<typename T> struct ComplexTraits<std::complex<T> >
{
	typedef std::complex<T> ValueType;
};

template<int N, typename T> struct ComplexTraits<nTuple<N, T> >
{
	typedef nTuple<N, typename ComplexTraits<T>::ValueType> ValueType;
};

template<typename T> struct ValueTraits
{
	typedef T ValueType;
};

template<int I, typename TG, typename TE> struct ValueTraits<Field<TG, I, TE> >
{
	typedef typename Field<TG, I, TE>::ValueType ValueType;
};

template<typename T> struct nTupleValueTraits
{
	typedef T ValueType;
};

template<int N, typename TE> struct nTupleValueTraits<nTuple<N, TE> >
{
	typedef typename nTuple<N, TE>::ValueType ValueType;
};


template<typename TL, typename TR>
struct OpMultiplication
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::ValueType,
			typename ValueTraits<TR>::ValueType>::ValueType ValueType;
}
;
template<typename TL, typename TR>
struct OpDivision
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::ValueType,
			typename ValueTraits<TR>::ValueType>::ValueType ValueType;
}
;
template<typename TL, typename TR>
struct OpAddition
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::ValueType,
			typename ValueTraits<TR>::ValueType>::ValueType ValueType;
}
;
template<typename TL, typename TR>
struct OpSubtraction
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::ValueType,
			typename ValueTraits<TR>::ValueType>::ValueType ValueType;
}
;
template<typename TL>
struct OpNegative
{
	typedef typename ValueTraits<TL>::ValueType ValueType;
}
;

} //namespace  _impl
} // namespace simpla

#endif /* TYPECONVERT_H_ */
