/*
 * calculus.h
 *
 *  Created on: 2014年9月23日
 *      Author: salmon
 */

#ifndef CALCULUS_H_
#define CALCULUS_H_

namespace simpla
{
template<typename TM, unsigned int IFORM> class Domain;
template<typename TD, typename TF> class Field;

struct HodgeStar
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef Domain<TM, TM::ndims - IFORM> type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef Field<typename domain_traits<TD>::type,
				UniOp<HodgeStar, Field<TD, TExpr>> > type;

	};

};

struct InteriorProduct
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef Domain<TM, IFORM - 1> type;
	};

	template<typename TM> struct domain_traits<Domain<TM, 0>>
	{
		typedef Zero type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef typename domain_traits<TD>::type domain_type;

		typedef typename std::conditional<
				std::is_same<Zero, domain_type>::value, Zero,
				Field<domain_type, UniOp<HodgeStar, Field<TD, TExpr>> >> ::type type;

	};

};

class Wedge
{
	template<typename, typename > struct domain_traits;

	template<typename TM, unsigned int IL, unsigned int IR> struct domain_traits<
			Domain<TM, IL>, Domain<TM, IR>>
	{
		typedef typename std::conditional<(IL + IR > TM::ndims), Zero,
				Domain<TM, IL + IR> >::type type;
	};

	template<typename, typename > struct field_traits;

	template<typename TDL, typename TL, typename TDR, typename TR>
	struct field_traits<Field<TDL, TL>, Field<TDR, TR>>
	{
		typedef typename domain_traits<TDL, TDR>::type domain_type;

		typedef typename std::conditional<
				std::is_same<Zero, domain_type>::value, Zero,
				Field<domain_type, BiOp<Wedge, Field<TDL, TL>, Field<TDL, TR>> >> ::type type;

	};

};

class ExteriorDerivative
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef typename std::conditional<(IFORM == 0), Zero,
				Domain<TM, IFORM - 1> >::type type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef Field<typename domain_traits<TD>::type,
				UniOp<ExteriorDerivative, Field<TD, TExpr>> > type;

	};
};

class CodifferentialDerivative
{
	template<typename > struct domain_traits;

	template<typename TM, unsigned int IFORM> struct domain_traits<
			Domain<TM, IFORM>>
	{
		typedef typename std::conditional<(IFORM == 0), Zero,
				Domain<TM, IFORM - 1> >::type type;
	};

	template<typename > struct field_traits;

	template<typename TD, typename TExpr>
	struct field_traits<Field<TD, TExpr>>
	{
		typedef Field<typename domain_traits<TD>::type,
				UniOp<CodifferentialDerivative, Field<TD, TExpr>> > type;

	};
};

}  // namespace simpla

#endif /* CALCULUS_H_ */
