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

namespace _impl
{
// NOTE: this is a replacement before C++11 available for CUDA


template<typename, typename > struct TypeConvertTraits;

template<>
struct TypeConvertTraits<int, int>
{
	typedef int Value;
};

template<>
struct TypeConvertTraits<long, long>
{
	typedef long Value;
};

template<>
struct TypeConvertTraits<double, double>
{
	typedef double Value;
};

template<>
struct TypeConvertTraits<std::complex<double>, std::complex<double> >
{
	typedef std::complex<double> Value;
};


#define TYPE_OP_RULE(TL,TR,TV) \
template<> struct TypeConvertTraits<TL, TR> { 	typedef TV Value; }; \
template<> struct TypeConvertTraits<TR, TL> { 	typedef TV Value; };

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
	typedef std::complex<T> Value;
};

template<typename T> struct ComplexTraits<std::complex<T> >
{
	typedef std::complex<T> Value;
};

template<int N, typename T> struct ComplexTraits<nTuple<N, T> >
{
	typedef nTuple<N, typename ComplexTraits<T>::Value> Value;
};

template<typename T> struct ValueTraits
{
	typedef T Value;
};

template<typename T> struct nTupleValueTraits
{
	typedef T Value;
};

template<int N, typename TE> struct nTupleValueTraits<nTuple<N, TE> >
{
	typedef typename nTuple<N, TE>::Value Value;
};


template<typename TL, typename TR>
struct OpMultiplication
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::Value,
			typename ValueTraits<TR>::Value>::Value Value;
}
;
template<typename TL, typename TR>
struct OpDivision
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::Value,
			typename ValueTraits<TR>::Value>::Value Value;
}
;
template<typename TL, typename TR>
struct OpAddition
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::Value,
			typename ValueTraits<TR>::Value>::Value Value;
}
;
template<typename TL, typename TR>
struct OpSubtraction
{
	typedef typename TypeConvertTraits<
			typename ValueTraits<TL>::Value,
			typename ValueTraits<TR>::Value>::Value Value;
}
;
template<typename TL>
struct OpNegative
{
	typedef typename ValueTraits<TL>::Value Value;
}
;

template<typename TL, typename TR> class OpEquality;

template<typename TL, typename TR> class OpLessThan;

} //namespace  _impl
} // namespace simpla

#endif /* TYPECONVERT_H_ */
