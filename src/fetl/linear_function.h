/*
 * linear_function.h
 *
 *  Created on: 2013-7-11
 *      Author: salmon
 */

#ifndef LINEAR_FUNCTION_H_
#define LINEAR_FUNCTION_H_

#include "expression.h"
namespace simpla
{
/**
 * A.x - b=0
 *
 *
 * */
template<typename TL> struct LinearFunction;

template<typename T> struct is_LinearFunction
{
	static const bool value = false;
};

template<typename T> struct is_LinearFunction<LinearFunction<T> >
{
	static const bool value = true;
};
template<> struct LinearFunction<NullType>
{
	typedef LinearFunction<NullType> ThisType;

	size_t idx;

	LinearFunction(size_t i) :
			idx(i)
	{
	}
	LinearFunction(ThisType const& rhs) = default;
	~LinearFunction()
	{

	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		try
		{
			v.at(idx) += a;
		} catch (...)
		{
			v.insert(std::make_pair(idx, a));
		}
	}
};

typedef LinearFunction<NullType> PlaceHolder;

struct PlaceHolderGenerator
{
	PlaceHolderGenerator(size_t)
	{
	}
	~PlaceHolderGenerator()
	{
	}

	PlaceHolder operator[](size_t s)
	{
		return (PlaceHolder(s));
	}

};

template<typename TL, typename TR>
struct LinearFunction<BiOp<OpPlus, LinearFunction<TL>, LinearFunction<TR> > >
{
	typedef LinearFunction<BiOp<OpPlus, LinearFunction<TL>, LinearFunction<TR> > > ThisType;
	typedef LinearFunction<TL> T1;
	typedef LinearFunction<TR> T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, a);
		r_.get_coeffs(v, b, a);
	}
}
;

template<typename TL, typename TR>
struct LinearFunction<BiOp<OpPlus, TL, LinearFunction<TR> > >
{
	typedef LinearFunction<BiOp<OpPlus, TL, LinearFunction<TR> > > ThisType;
	typedef TL T1;
	typedef LinearFunction<TR> T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		b += l_ * a;
		r_.get_coeffs(v, b, a);
	}

};
template<typename TL, typename TR>
struct LinearFunction<BiOp<OpPlus, LinearFunction<TL>, TR> >
{
	typedef LinearFunction<BiOp<OpPlus, LinearFunction<TL>, TR> > ThisType;
	typedef LinearFunction<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, a);
		b += r_ * a;
	}

};

template<typename TL, typename TR>
struct LinearFunction<BiOp<OpMinus, LinearFunction<TL>, LinearFunction<TR> > >
{
	typedef LinearFunction<
			BiOp<OpMinus, LinearFunction<TL>, LinearFunction<TR> > > ThisType;
	typedef LinearFunction<TL> T1;
	typedef LinearFunction<TR> T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, a);
		r_.get_coeffs(v, b, -a);
	}

};

template<typename TL, typename TR>
struct LinearFunction<BiOp<OpMinus, TL, LinearFunction<TR> > >
{
	typedef LinearFunction<BiOp<OpMinus, TL, LinearFunction<TR> > > ThisType;
	typedef TL T1;
	typedef LinearFunction<TR> T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		b += l_ * a;
		r_.get_coeffs(v, b, -a);
	}

};
template<typename TL, typename TR>
struct LinearFunction<BiOp<OpMinus, LinearFunction<TL>, TR> >
{
	typedef LinearFunction<BiOp<OpMinus, LinearFunction<TL>, TR> > ThisType;
	typedef LinearFunction<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, a);
		b -= r_ * a;
	}

};

template<typename TL>
struct LinearFunction<UniOp<OpNegate, LinearFunction<TL> > >
{
	typedef LinearFunction<UniOp<OpNegate, LinearFunction<TL> > > ThisType;
	typedef LinearFunction<TL> T1;

	T1 l_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l) :
			l_(l)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, -a);
	}

};
template<typename TL, typename TR>
struct LinearFunction<BiOp<OpMultiplies, LinearFunction<TL>, TR> >
{
	typedef LinearFunction<BiOp<OpMultiplies, LinearFunction<TL>, TR> > ThisType;
	typedef LinearFunction<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, a * r_);
	}

};

template<typename TL, typename TR>
struct LinearFunction<BiOp<OpMultiplies, TL, LinearFunction<TR> > >
{
	typedef LinearFunction<BiOp<OpMultiplies, TL, LinearFunction<TR> > > ThisType;
	typedef TL T1;
	typedef LinearFunction<TR> T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		r_.get_coeffs(v, b, l_ * a);
	}

};

template<typename TL, typename TR>
struct LinearFunction<BiOp<OpDivides, LinearFunction<TL>, TR> >
{
	typedef LinearFunction<BiOp<OpDivides, LinearFunction<TL>, TR> > ThisType;
	typedef LinearFunction<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearFunction(ThisType const& rhs) = default;

	LinearFunction(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void get_coeffs(TV & v, TB &b) const
	{
		get_coeffs(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void get_coeffs(TV & v, TB &b, TA const & a) const
	{
		l_.get_coeffs(v, b, a / r_);
	}

};

template<typename TL, typename TR> inline auto operator*(TL const &lhs,
		LinearFunction<TR> const& rhs)
				DECL_RET_TYPE((LinearFunction<BiOp<OpMultiplies,TL,LinearFunction<TR > > >(lhs ,rhs)))

template<typename TL, typename TR> inline auto operator*(
		LinearFunction<TL> const& lhs,
		TR const& rhs)
				DECL_RET_TYPE((LinearFunction<BiOp<OpMultiplies, LinearFunction<TL >,TR > >(lhs ,rhs)))

template<typename TL, typename TR> inline auto operator/(
		LinearFunction<TL> const& lhs,
		TR const& rhs)
				DECL_RET_TYPE((LinearFunction<BiOp<OpDivides, LinearFunction<TL >,TR > >(lhs ,rhs)))

template<typename TL, typename TR> void operator*(LinearFunction<TL> const& l,
		LinearFunction<TR> const& r) = delete;

template<typename TL, typename TR> void operator/(LinearFunction<TL> const& l,
		LinearFunction<TR> const& r) = delete;

template<typename TL>
inline auto operator-(LinearFunction<TL> const & expr)
DECL_RET_TYPE(( LinearFunction<UniOp<OpNegate,LinearFunction<TL> > >(expr)))

template<typename TL, typename TR> inline auto operator+(
		LinearFunction<TL> const& l, LinearFunction<TR> const& r)
		DECL_RET_TYPE(( LinearFunction<BiOp<OpPlus,LinearFunction<TL>,
						LinearFunction< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator+(TL const& l,
		LinearFunction<TR> const& r)
				DECL_RET_TYPE(( LinearFunction<BiOp<OpPlus,TL, LinearFunction< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator+(
		LinearFunction<TL> const& l,
		TR const& r)
				DECL_RET_TYPE(( LinearFunction<BiOp<OpPlus,LinearFunction<TL>, TR > > (l,r)))

template<typename TL, typename TR> inline auto operator-(
		LinearFunction<TL> const& l, LinearFunction<TR> const& r)
		DECL_RET_TYPE(( LinearFunction<BiOp<OpMinus,LinearFunction<TL>,
						LinearFunction< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator-(TL const& l,
		LinearFunction<TR> const& r)
				DECL_RET_TYPE(( LinearFunction<BiOp<OpMinus,TL, LinearFunction< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator-(
		LinearFunction<TL> const& l,
		TR const& r)
				DECL_RET_TYPE(( LinearFunction<BiOp<OpMinus,LinearFunction<TL>, TR > > (l,r)))

template<typename TL, typename TR> inline auto operator==(
		LinearFunction<TL> const& l, LinearFunction<TR> const& r)
		DECL_RET_TYPE((l-r))

template<typename TL, typename TR> inline auto operator==(TL const& l,
		LinearFunction<TR> const& r) DECL_RET_TYPE(l-r)

template<typename TL, typename TR> inline auto operator==(
		LinearFunction<TL> const& l, TR const& r) DECL_RET_TYPE((l-r))

}
// namespace simpla

#endif /* LINEAR_FUNCTION_H_ */
