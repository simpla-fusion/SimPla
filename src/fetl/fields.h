/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: fetl_fields.h 1009 2011-02-07 23:20:45Z salmon $
 * fetl_fields.h
 *
 * Created on: 2009-3-31
 *  Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */

#ifndef FETL_DETAIL_FIELD_H_
#define FETL_DETAIL_FIELD_H_
#include <typeinfo>

#include "include/simpla_defs.h"
#include "fetl/fetl_defs.h"
#include "engine/arrayobject.h"
#include "engine/datatype.h"
#include "typeconvert.h"

namespace simpla
{

/**
 * Field
 * Expression Template of field or differential form
 *
 *  Semantics:
 *   Define the abstract  rules of fields' arithmetic and Vector calculus.
 *   All specific calculus are defined in "Grid".
 *
 */
template<typename TG, int IFORM, typename TV>
struct Field: public ArrayObject
{
public:
	typedef TV ValueType;

	typedef TG Grid;

	enum
	{
		IForm = IFORM
	};

	typedef Field<Grid, IForm, ValueType> ThisType;

	typedef ThisType const &ConstReference;

	typedef TR1::shared_ptr<ThisType> Holder;

	Grid const & grid;

	Field(Grid const & pgrid) :
			ArrayObject(DataType<ValueType>(), pgrid.get_field_shape(IForm)), grid(
					pgrid)
	{
	}

	virtual ~Field()
	{
	}

// Interpolation  ----------------------------------------------------------------------

//	inline ValueType operator()(RVec3 const & x)
//	{
//		return (grid.Gather(*this, x));
//	}
//	inline void Add(RVec3 const & x, ValueType const & v)
//	{
//		grid.Scatter(*this, x, v);
//	}

// Assignment --------

	inline void Add(ThisType const &rhs)
	{
		Grid::Add(*this, rhs);
	}

	inline ThisType & operator =(ThisType const &rhs)
	{
		if (!(this->IsSame(rhs)))
		{
			grid.Assign(*this, rhs);
		}
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator =(Field<Grid, IForm, TR> const &rhs)
	{
		grid.Assign(*this, rhs);
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator +=(Field<Grid, IForm, TR> const &rhs)
	{
		*this = *this + rhs;
		return (*this);
	}
	template<typename TR>
	inline ThisType & operator -=(Field<Grid, IForm, TR> const &rhs)
	{
		*this = *this - rhs;
		return (*this);
	}
	inline ThisType & operator *=(Real rhs)
	{
		*this = *this * rhs;
		return (*this);
	}
	inline ThisType & operator /=(Real rhs)
	{
		*this = *this / rhs;
		return (*this);
	}
	template<typename TR>
	inline ThisType & operator =(TR const &rhs)
	{
		if (CheckEquationHasVariable(rhs, *this))
		{
			ThisType tmp(grid);
			grid.Assign(tmp, rhs);
			grid.Assign(*this, tmp);
		}
		else
		{
			grid.Assign(*this, rhs);
		}

		return (*this);
	}

	bool IsSame(ThisType const & rhs) const
	{
		return (ArrayObject::get_data() == rhs.get_data());
	}

	virtual inline bool CheckType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(ThisType));
	}
	virtual inline bool CheckValueType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(ValueType));
	}

	//----------------------------------------------------------------------

	inline void Add(size_t s, ValueType const & v)
	{
//#pragma omp atomic
		ArrayObject::value<ValueType>(s) += v;
	}

	inline ValueType & operator[](size_t s)
	{
		return (ArrayObject::value<ValueType>(s));
	}

	inline ValueType const &operator[](size_t s) const
	{
		return (ArrayObject::value<ValueType>(s));
	}

	static const Field<TG, IForm, Int2Type<0> > ZERO;
	static const Field<TG, IForm, Int2Type<1> > ONE;

};

//------------------------------------------------------------------------------------------
template<typename TG, int IFORM, int IFORM2, typename TLExpr,
		template<typename > class TOP>
struct Field<TG, IFORM, TOP<Field<TG, IFORM2, TLExpr> > >
{
	typedef Field<TG, IFORM2, TLExpr> TL;

	typedef Field<TG, IFORM, TOP<Field<TG, IFORM2, TLExpr> > > ThisType;

	typedef typename Field<TG, IFORM2, TLExpr>::ValueType ValueType;

	typedef TG Grid;

	enum
	{
		IForm = IFORM
	};

	typename simpla::_impl::TypeTraits<TL>::ConstReference lhs_;

	Grid const &grid;

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return grid.eval(*this, s);
	}

};

template<typename TG, int IFORM, typename TL, typename TR, template<typename,
		typename > class TOP>
struct Field<TG, IFORM, TOP<TL, TR> >
{
	enum
	{
		IForm = IFORM
	};
	typedef Field<TG, IFORM, TOP<TL, TR> > ThisType;

	typedef typename TOP<TL, TR>::ValueType ValueType;

	typename _impl::TypeTraits<TL>::ConstReference lhs_;
	typename _impl::TypeTraits<TR>::ConstReference rhs_;

	typedef TG Grid;
	Grid const & grid;

	Field(TL const &lhs, TR const & rhs) :
			grid(selectGrid(lhs, rhs)), lhs_(lhs), rhs_(rhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return grid.eval(*this, s);
	}
private:

	template<int IFORML, typename TLExpr, int IFORMR, typename TRExpr>
	static inline Grid const &selectGrid(Field<Grid, IFORML, TLExpr> const &l,
			Field<Grid, IFORMR, TRExpr> const &r)
	{
		return l.grid;
	}

	template<typename TVL, int IFORMR, typename TRExpr>
	static inline Grid const &selectGrid(TVL const &l,
			Field<Grid, IFORMR, TRExpr> const &r)
	{
		return r.grid;
	}

	template<int IFORML, typename TLExpr, typename TVR>
	static inline Grid const &selectGrid(Field<Grid, IFORML, TLExpr> const &l,
			TVR const &r)
	{
		return l.grid;
	}

};

} //namespace simpla
#endif  // FETL_DETAIL_FIELD_H_
