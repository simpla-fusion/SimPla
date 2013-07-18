/*Copyright (C) 2007-2013 YU Zhi. All rights reserved.
 *
 * $Id: field.h 1009 2011-02-07 23:20:45Z salmon $
 * field.h
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

#ifndef FIELD_H_
#define FIELD_H_

#include <stddef.h> // for NULL
#include "engine/object.h"
namespace simpla
{

/**
 * Field
 * Expression Template of field or differential form
 *
 *  Semantics:
 *   Define the abstract  rules of fields' arithmetic and Vector calculus.
 *   All specific calculus are implemented in "Grid".
 *
 */

template<typename TG, int IFORM, typename TV>
struct Field: public Object
{
public:

	typedef TG Grid;

	typedef TV Value;

	typedef typename Grid::Index Index;

	typedef typename Grid::Coordinates Coordinates;

	enum
	{
		IForm = IFORM
	};

	typedef Field<Grid, IForm, Value> ThisType;

	typedef ThisType const &ConstReference;

	const Grid & grid;

	const size_t value_size_in_bytes;

	typename Grid::Storage storage;

public:

	Field(Grid const & pgrid, size_t value_size = sizeof(Value)) :
			storage(), grid(pgrid), value_size_in_bytes(value_size)
	{
		Init();
	}

	virtual ~Field()
	{
	}

	void Init()
	{
		grid.InitEmptyField(this);
	}

	bool CheckType(std::type_info const &rhs) const
	{
		return (typeid(ThisType) == rhs);
	}

	bool IsEmpty() const
	{
		return (storage == typename Grid::Storage());
	}
	bool IsSame(ThisType const & rhs) const
	{
		return (storage == rhs.storage);
	}
// Assignment --------

	template<typename TR>
	inline ThisType & operator =(TR const &rhs)
	{
		Init();
//
//		if (CheckEquationHasVariable(rhs, *this))
//		{
//			ThisType tmp(grid);
//			grid.Assign(tmp, rhs);
//			grid.Assign(*this, tmp);
//		}
//		else
		{
			grid.Assign(*this, rhs);
		}

		return (*this);
	}

	inline ThisType & operator =(ThisType const &rhs)
	{
		Init();

		if (!(this->IsSame(rhs)))
		{
			grid.Assign(*this, rhs);
		}
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator =(Field<Grid, IForm, TR> const &rhs)
	{
		Init();
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

	//----------------------------------------------------------------------

	inline Value & operator[](Index s)
	{
		return (grid.GetValue(*this, s));
	}

	inline Value const &operator[](Index s) const
	{
		return (grid.GetConstValue(*this, s));
	}

	// interpolate
	inline Value operator()(Coordinates x) const
	{
		return (grid.GetValue(*this, x));
	}

	inline void Add(Coordinates x, Value const & v)
	{
		//#pragma omp atomic
		grid.AddValue(*this, x, v);
	}

//
//	template<typename TG, typename TL, int IR, typename TRExpr>
//	inline bool CheckEquationHasVariable(TL const & eqn,
//			Field<TG, IR, TRExpr> const & v)
//	{
//		return false;
//	}
//	template<typename TG, int IL, typename TLExpr, typename TRExpr>
//	inline bool CheckEquationHasVariable(Field<TG, IL, TLExpr> const & eqn,
//			Field<TG, IL, TRExpr> const & v)
//	{
//		return (eqn.IsSame(v));
//	}
//
//	template<typename TG, typename TLExpr, int IL, int IR, typename TRExpr,
//			template<typename > class TOP>
//	inline bool CheckEquationHasVariable(
//			Field<TG, IL, TOP<TLExpr> > const & eqn,
//			Field<TG, IR, TRExpr> const & v)
//	{
//		return CheckEquationHasVariable(eqn.lhs_, v);
//	}
//
//	template<typename TG, typename TL, typename TR, int IL, int IR,
//			typename TRExpr, template<typename, typename > class TOP>
//	inline bool CheckEquationHasVariable(
//			Field<TG, IL, TOP<TL, TR> > const & eqn,
//			Field<TG, IR, TRExpr> const & v)
//	{
//		return CheckEquationHasVariable(eqn.lhs_, v)
//				|| CheckEquationHasVariable(eqn.rhs_, v);
//	}
//
//	template<int IFORM, typename TG, typename TExpr>
//	TR1::shared_ptr<Field<TG, IFORM, typename Field<TG, IFORM, TExpr>::Grid> > //
//	DuplicateField(Field<TG, IFORM, TExpr> const f)
//	{
//		typedef typename Field<TG, IFORM, TExpr>::Grid Grid;
//
//		return (TR1::shared_ptr<Field<TG, IFORM, Grid> >(
//				new Field<TG, IFORM, Grid>(f.grid)));
//	}

};

//------------------------------------------------------------------------------------------

/// UniOperator
template<typename TG, int IFORM, int IFORM2, typename TLExpr,
		template<typename > class TOP>
struct Field<TG, IFORM, TOP<Field<TG, IFORM2, TLExpr> > >
{
	typedef Field<TG, IFORM2, TLExpr> TL;

	typedef Field<TG, IFORM, TOP<Field<TG, IFORM2, TLExpr> > > ThisType;

	typedef typename Field<TG, IFORM2, TLExpr>::Value Value;

	typedef TG Grid;

	typedef typename Grid::Index Index;

	enum
	{
		IForm = IFORM
	};

	typename TypeTraits<TL>::ConstReference lhs_;

	Grid const &grid;

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline Value operator[](Index s) const
	{
		return (grid.eval(*this, s));
	}

};

/// BiOperator
template<typename TG, int IFORM, typename TL, typename TR, template<typename,
		typename > class TOP>
struct Field<TG, IFORM, TOP<TL, TR> >
{
	enum
	{
		IForm = IFORM
	};
	typedef Field<TG, IFORM, TOP<TL, TR> > ThisType;

	typedef typename TOP<TL, TR>::Value Value;

	typename TypeTraits<TL>::ConstReference lhs_;
	typename TypeTraits<TR>::ConstReference rhs_;

	typedef TG Grid;

	typedef typename Grid::Index Index;

	Grid const & grid;

	Field(TL const &lhs, TR const & rhs) :
			grid(selectGrid(lhs, rhs)), lhs_(lhs), rhs_(rhs)
	{
	}

	inline Value operator[](Index s) const
	{
		return (grid.eval(*this, s));
	}
private:

	template<int IFORML, typename TLExpr, int IFORMR, typename TRExpr>
	static inline Grid const &selectGrid(Field<Grid, IFORML, TLExpr> const &l,
			Field<Grid, IFORMR, TRExpr> const &r)
	{
		return (l.grid);
	}

	template<typename TVL, int IFORMR, typename TRExpr>
	static inline Grid const &selectGrid(TVL const &l,
			Field<Grid, IFORMR, TRExpr> const &r)
	{
		return (r.grid);
	}

	template<int IFORML, typename TLExpr, typename TVR>
	static inline Grid const &selectGrid(Field<Grid, IFORML, TLExpr> const &l,
			TVR const &r)
	{
		return (l.grid);
	}

};

} // namespace simpla
#endif  // FIELD_H_
