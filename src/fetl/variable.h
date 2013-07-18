/*
 * variable.h
 *
 *  Created on: 2013-7-11
 *      Author: salmon
 */

#ifndef VARIABLE_H_

#include "fetl_defs.h"
#include "primitives/typeconvert.h"

#ifdef DEBUG
#include "utilities/log.h"
#endif

namespace simpla
{

#ifndef VARIABLE_EXP_WIDTH
#define MAX_VARIABLE_EXP_WIDTH 10
#endif

template<typename TI = size_t, typename TV = NullType>
struct Variable
{
	typedef TI Index;

	Index idx;

	template<typename TV>
	int assign(TI * id, TV * d, int s,
			const int MAX = MAX_VARIABLE_EXP_WIDTH) const
	{
		int i = 0;
		while (i < s)
		{
			if (id[i] == idx)
			{
				break;
			}
			++i;
		}
#ifdef DEBUG
		CHECK(i < MAX_VARIABLE_EXP_WIDTH);
#endif

		d[i] = (i = s) ? 1.0 : (d[i] + 1.0);

		return (i);

	}
};
template<typename TI, typename TL, typename TR>
struct Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >
{
	typedef TI Index;
	TL l
	Variable<TI, TR> const & r;

	template<typename TV>
	inline int assign(TI * id, TV * d, int s, const int MAX =
			MAX_VARIABLE_EXP_WIDTH) const
	{
		s = r.assign(id, d, s, MAX);

		for (int i = 0; i < s; ++i)
		{
			d[i] *= l;
		}

		return (s);

	}
};

template<typename TI, typename TL, typename TR>
struct Variable<TI, _impl::OpAddition<Variable<TI, TL>, Variable<TI, TR> > >
{
	typedef TI Index;
	Variable<TI, TL> const & l;
	Variable<TI, TR> const & r;

	template<typename TV>
	inline int assign(TI * id, TV * d, int s, const int MAX =
			MAX_VARIABLE_EXP_WIDTH) const
	{
		s = r.assign(id, d, s, MAX);
		s = l.assign(id, d, s, MAX);

		return (s);
	}
};
template<typename TI, typename TL, typename TR>
Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >                 //
operator*(TL a, Variable<TI, TR> const & v)
{
	return (Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >(a, v));
}

template<typename TI, typename TL, typename TR>
Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >                 //
operator*(Variable<TI, TR> const & v, TL a)
{
	return (Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >(a, v));
}
template<typename TI, typename TL, typename TR>
Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >                 //
operator/(Variable<TI, TR> const & v, TL a)
{
	return (Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > >(
			1.0 / a, v));
}
template<typename TI, typename TL, typename TR>
Variable<TI, _impl::OpAddition<Variable<TI, TL>, Variable<TI, TR> > >         //
operator+(Variable<TI, TL> vl, Variable<TI, TR> const & vr)
{
	return (Variable<TI, _impl::OpAddition<Variable<TI, TL>, Variable<TI, TR> > >(
			vl, vr));
}

template<typename TI, typename TL, typename TR>
Variable<TI,
		_impl::OpAddition<Variable<TI, TL>,
				Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > > > >  //
operator-(Variable<TI, TL> vl, Variable<TI, TR> const & vr)
{
	return (Variable<TI,
			_impl::OpAddition<Variable<TI, TL>,
					Variable<TI, _impl::OpMultiplication<TL, Variable<TI, TR> > > > >(
			vl, -1.0 * vr));
}

template<typename TG, int IFORM>
struct Field<TG, IFORM, Variable<typename TG::Index, NullType> >
{
public:

	typedef TG Grid;

	typedef typename Grid::Index Index;

	typedef typename Grid::Coordinates Coordinates;

	typedef Variable<Index, NullType> Value;

	enum
	{
		IForm = IFORM
	};

	typedef Field<Grid, IForm, Value> ThisType;

	typedef ThisType const &ConstReference;

	const Grid & grid;

public:

	Field(Grid const & pgrid) :
			grid(pgrid)
	{
	}

	~Field()
	{
	}

	inline Value const &operator[](Index s) const
	{
		Value res =
		{ s, 1.0 };
		return (res);
	}

private:
	template<typename TR> inline ThisType & operator =(TR const &rhs);
};

}  // namespace simpla

#endif /* VARIABLE_H_ */
