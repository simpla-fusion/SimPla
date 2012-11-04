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
#include "typeconvert.h"
#include "fetl_defs.h"

#include "include/simpla_defs.h"

#include "utilities/properties.h"
#include "datastruct/ndarray.h"
#include "engine/datatype.h"
#include "engine/context.h"
#include "io/read_hdf5.h"
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
struct Field: public NdArray
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
			NdArray(DataType<ValueType>(), pgrid.get_field_shape(IForm)), grid(
					pgrid)
	{
	}

	virtual ~Field()
	{
	}

	static TR1::shared_ptr<ThisType> Create(Context<TG> * ctx, ptree const & pt)
	{

		TR1::shared_ptr<ThisType> res(new ThisType(ctx->grid));

		boost::optional<ptree const &> o_pt = pt.get_child_optional(
				"<xmlattr>");

		if (!!o_pt)
		{
			res->properties = *o_pt;

			LOG << "Load Field ["

			<< res->properties.template get<std::string>("Name") << ":"

			<< res->properties.template get<std::string>("Type") << "]";
		}

		boost::optional<ptree const &> value = pt.get_child_optional("Value");

		if (!value)
		{
			res->Clear();
		}
		else if (value->get<std::string>("<xmlattr>.Format") == "HDF")
		{
			std::string url = value->get_value<std::string>();
			io::ReadData(url, res);
		}
		else if (value->get<std::string>("<xmlattr>.Format") == "XML")
		{
			if (IFORM == IOneForm || IFORM == ITwoForm)
			{
				*res = value->get_value<nTuple<THREE, ValueType> >(
						pt_trans<nTuple<THREE, ValueType>,
								typename ptree::key_type>());
			}
			else
			{
				*res = value->get_value<ValueType>(
						pt_trans<ValueType, typename ptree::key_type>());
			}
		}

		return res;
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
		return (NdArray::get_data() == rhs.get_data());
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
		NdArray::value<ValueType>(s) += v;
	}

	inline ValueType & operator[](size_t s)
	{
		return (NdArray::value<ValueType>(s));
	}

	inline ValueType const &operator[](size_t s) const
	{
		return (NdArray::value<ValueType>(s));
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

	typename _impl::TypeTraits<TL>::ConstReference lhs_;

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
