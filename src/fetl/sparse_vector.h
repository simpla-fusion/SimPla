/*
 * sparse_vector.h
 *
 *  Created on: 2013-7-11
 *      Author: salmon
 */

#ifndef SPARSE_VECTOR
#define SPARSE_VECTOR

#include "fetl_defs.h"
#include "primitives/typeconvert.h"
#include <map>

namespace simpla
{

template<typename TV>
struct SparseVector:public std::map<size_t, TV>
{
	typedef TV Value;
};

template<>
struct SparseVector<NullType>
{

	size_t idx;

	template<typename TV>
	inline void assign(std::vector<size_t> * id, std::vector<TV> * d) const
	{
		typename std::vector<size_t>::iterator it1 = id->begin();
		typename std::vector<TV>::iterator it2 = d->begin();

		while (it1 != id->end())
		{
			if (*it1 == idx)
			{
				break;
			}
			++it1;
			++it2;
		}

		if (it1 == id->end())
		{
			d->push_back(1.0);
		}
		else
		{
			*it2 += 1.0;
		}

	}
};
template<typename TL, typename TR>
struct SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >
{
	size_t Index;
	TL l
	SparseVector<TR> const & r;

	template<typename TV>
	inline void assign(std::vector<size_t> * id, std::vector<TV> * d) const
	{
		r.assign(id, d);
		typename std::vector<size_t>::iterator it1 = id->begin();
		typename std::vector<TV>::iterator it2 = d->begin();
		for (; it1 != id->end(); ++it1, ++it2)
		{
			*it2 *= l;
		}

	}
};

template<typename TL, typename TR>
struct SparseVector<_impl::OpAddition<SparseVector<TL>, SparseVector<TR> > >
{
	size_t Index;
	SparseVector<TL> const & l;
	SparseVector<TR> const & r;

	template<typename TV>
	inline void assign(std::vector<size_t> * id, std::vector<TV> * d) const
	{
		r.assign(id, d);
		l.assign(id, d);

	}
};
template<typename TL, typename TR>
SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >                 //
operator*(TL a, SparseVector<TR> const & v)
{
	return (SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >(a, v));
}

template<typename TL, typename TR>
SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >                 //
operator*(SparseVector<TR> const & v, TL a)
{
	return (SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >(a, v));
}
template<typename TL, typename TR>
SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >                 //
operator/(SparseVector<TR> const & v, TL a)
{
	return (SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > >(
			1.0 / a, v));
}
template<typename TL, typename TR>
SparseVector<_impl::OpAddition<SparseVector<TL>, SparseVector<TR> > >         //
operator+(SparseVector<TL> vl, SparseVector<TR> const & vr)
{
	return (SparseVector<_impl::OpAddition<SparseVector<TL>, SparseVector<TR> > >(
			vl, vr));
}

template<typename TL, typename TR>
SparseVector<
		_impl::OpAddition<SparseVector<TL>,
				SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > > > >  //
operator-(SparseVector<TL> vl, SparseVector<TR> const & vr)
{
	return (SparseVector<
			_impl::OpAddition<SparseVector<TL>,
					SparseVector<_impl::OpMultiplication<TL, SparseVector<TR> > > > >(
			vl, -1.0 * vr));
}

namespace _impl
{

template<typename TL, typename TR>
struct TypeConvertTraits<TL, SparseVector<TR> >
{
	typedef std::complex<double> Value;
};

}  // namespace _impl

template<typename TG, int IFORM>
struct Field<TG, IFORM, SparseVector<NullType> >
{
public:

	typedef TG Grid;

	typedef typename Grid::Index Index;

	typedef typename Grid::Coordinates Coordinates;

	typedef SparseVector<NullType> Value;

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
		Value res;
		res.idx = s;
		return (res);
	}

private:
	template<typename TR> inline ThisType & operator =(TR const &rhs);
};

template<typename TG, int IFORM, typename TV>
struct Field<TG, IFORM, SparseVector<TV> >
{
public:

	typedef TG Grid;

	typedef typename Grid::Index Index;

	typedef typename Grid::Coordinates Coordinates;

	typedef SparseVector<TV> Value;

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
		Value res;
		res.idx = s;
		return (res);
	}

private:
	template<typename TR> inline ThisType & operator =(TR const &rhs);
};

}  // namespace simpla

#endif /* SPARSE_VECTOR */
