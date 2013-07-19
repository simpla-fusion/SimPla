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

#ifndef GEOMETRY_FIELD_H_
#define GEOMETRY_FIELD_H_

#include <stddef.h> // for NULL
#include "engine/object.h"
#include <utility>
#include <type_traits>
#include "field.h"
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

template<typename TV, typename TGrid>
struct GeometryField: public Field<TV>
{
public:

	typedef TG Grid;

	typedef TV Value;

	typedef typename Grid::Index Index;

	typedef typename Grid::Coordinates Coordinates;

	typedef Field<TV> BaseType;

	enum
	{
		IForm = IFORM
	};

	typedef GeometryField<Value, Grid> ThisType;

	typedef ThisType const &ConstReference;

	const Grid & grid;

public:

	GeometryField(Grid const & pgrid, size_t value_size = sizeof(Value)) :
			grid(pgrid), BaseType(grid.get_num_of_elements(), value_size)
	{
		grid.InitEmptyField(this);
	}

	virtual ~GeometryField()
	{
	}

	bool CheckType(std::type_info const &rhs) const
	{
		return (typeid(ThisType) == rhs || BaseType::CheckType(rhs));
	}

// Assignment --------

	template<typename TR>
	inline ThisType & operator =(TR const &rhs)
	{

		grid.Assign(*this, rhs);

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
	inline ThisType & operator =(GeometryField<Grid, TR> const &rhs)
	{
		Init();
		grid.Assign(*this, rhs);
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator +=(GeometryField<Grid, TR> const &rhs)
	{
		*this = *this + rhs;
		return (*this);
	}
	template<typename TR>
	inline ThisType & operator -=(GeometryField<Grid, TR> const &rhs)
	{
		*this = *this - rhs;
		return (*this);
	}

	//----------------------------------------------------------------------
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

} // namespace simpla
#endif  // GEOMETRY_FIELD_H_
