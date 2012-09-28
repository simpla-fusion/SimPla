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
#include "primitives/typetraits.h"
#include "engine/object.h"

namespace simpla
{
namespace fetl
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

template<int IFORM, typename TV, typename TG>
struct Field: public Object
{
public:
	typedef TV ValueType;

	typedef TG Grid;

	static const int IForm = IFORM;

	typedef Field<IForm, ValueType, Grid> ThisType;

	typedef ThisType const &ConstReference;

	typedef typename ElementTypeTraits<ThisType>::Type ElementType;

	typedef TR1::shared_ptr<ThisType> Holder;

	Grid const & grid;

	Field(ThisType const & rhs) :
			grid(rhs.grid), num_of_elements_(rhs.num_of_elements_), //
			data_(rhs.data_)
	{
	}

	Field(Grid const & pgrid) :
			grid(pgrid), num_of_elements_(pgrid.get_num_of_elements(IForm)), //
			data_(Object::alloc<ValueType>(num_of_elements_))
	{
	}

	virtual ~Field()
	{
	}

// Interpolation  ----------------------------------------------------------------------

	inline ElementType operator()(RVec3 const & x)
	{
		return (grid.Gather(*this, x));
	}
	inline void Add(RVec3 const & x, ElementType const & v)
	{
		grid.Scatter(*this, x, v);
	}

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

	template<typename TRV, typename TR>
	inline ThisType & operator =(Field<IForm, TRV, TR> const &rhs)
	{
		grid.Assign(*this, rhs);
		return (*this);
	}

	template<typename TRV, typename TR>
	inline ThisType & operator +=(Field<IForm, TRV, TR> const &rhs)
	{
		*this = *this + rhs;
		return (*this);
	}
	template<typename TRV, typename TR>
	inline ThisType & operator -=(Field<IForm, TRV, TR> const &rhs)
	{
		*this = *this - rhs;
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
		return (data_.get() == rhs.data_.get());
	}

	virtual inline bool CheckType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(ThisType));
	}
	virtual inline bool CheckValueType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(ValueType));
	}

	virtual inline size_t get_element_size_in_bytes() const
	{
		return sizeof(ValueType);
	}

	virtual inline std::string get_element_type_desc() const
	{
		return "H5T_NATIVE_DOUBLE";
	}

	virtual inline void const * get_data() const
	{
		return reinterpret_cast<const void*>(data_.get());
	}

	virtual inline void * get_data()
	{
		return reinterpret_cast<void*>(data_.get());
	}

	virtual inline int get_dimensions(size_t* dims) const
	{
		return (grid.get_field_shape(dims, IForm));
	}

	virtual inline size_t get_size_in_bytes() const
	{
		return (num_of_elements_ * sizeof(ValueType));
	}

	virtual inline bool Empty() const
	{
		return (data_ == NULL);
	}

	//----------------------------------------------------------------------

	inline void Add(size_t s, ValueType const & v)
	{
#pragma omp atomic
		data_.get()[s] += v;
	}

	inline ValueType & operator[](size_t s)
	{
		return (data_.get()[s]);
	}

	inline ValueType const &operator[](size_t s) const
	{
		return (data_.get()[s]);
	}

	static const Field<IFORM, TV, Int2Type<0> > ZERO;
	static const Field<IFORM, TV, Int2Type<1> > ONE;
private:
	size_t num_of_elements_;
	TR1::shared_ptr<ValueType> data_;

};
template<int IFORM, typename TV, typename TG>
const Field<IFORM, TV, Int2Type<0> > Field<IFORM, TV, TG>::ZERO;
template<int IFORM, typename TV, typename TG>
const Field<IFORM, TV, Int2Type<1> > Field<IFORM, TV, TG>::ONE;

} // namespace fetl

template<int IFORM, typename TV, typename TG>
struct TypeTraits<fetl::Field<IFORM, TV, TG> >
{
	typedef fetl::Field<IFORM, TV, TG> & Reference;
	typedef const fetl::Field<IFORM, TV, TG> & ConstReference;
};
} //namespace simpla
#endif  // FETL_DETAIL_FIELD_H_
