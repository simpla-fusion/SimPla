/*
 * sparse_array.h
 *
 *  Created on: 2012-3-28
 *      Author: salmon
 */

#ifndef SPARSE_ARRAY_H_
#define SPARSE_ARRAY_H_

#include <iostream>

#include "include/simpla_defs.h"
#include "engine/object.h"
#include "primitives/operation.h"
#include "primitives/typetraits.h"
#include "primitives/ntuple.h"
namespace simpla
{

template<typename T, typename TExpr = NullType> class SparseArray;
template<typename T>
class SparseArray<T, NullType> : public std::map<size_t, T>
{
public:
	typedef T ValueType;
	typedef std::map<size_t, ValueType> BaseType;
	typedef SparseArray<T, NullType> ThisType;

	SparseArray(ThisType const & rhs) :
			BaseType(rhs.begin(), rhs.end())
	{
	}

	SparseArray()
	{
	}

	~SparseArray()
	{
	}

	// NOTE: The behaves of  reference and constant reference are not exactly same.

	inline ValueType & operator[](size_t s)
	{
		if (BaseType::find(s) == BaseType::end())
		{
			BaseType::operator[](s) = ZERO;
		}

		return (BaseType::operator[](s));
	}

	inline ValueType const& operator[](size_t s) const
	{
		typename std::map<size_t, ValueType>::const_iterator it =
				BaseType::find(s);
		return (it != BaseType::end()) ? it->second : ZERO;
	}

	template<typename TR, typename TRExpr>
	ThisType &operator=(SparseArray<TR, TRExpr> const &rhs)
	{
		BaseType t;
		for (typename SparseArray<TR, TRExpr>::iterator it = rhs.begin();
				it != rhs.end(); ++it)
		{
			t.insert(std::make_pair(it->first, rhs[it->first]));
		}
		t.swap(*this);
		return (*this);
	}

	static const ValueType ZERO;
};

template<typename T> typename SparseArray<T, NullType>::ValueType const //
SparseArray<T, NullType>::ZERO = 0;

template<> typename SparseArray<Vec3, NullType>::ValueType const //
SparseArray<Vec3, NullType>::ZERO =
{ 0, 0, 0 };

template<typename T, typename TExpr> std::ostream &
operator<<(std::ostream& os, const SparseArray<T, TExpr> & tv)
{
	for (typename SparseArray<T, TExpr>::const_iterator it = tv.begin();
			it != tv.end(); ++it)
	{
		os << "\t[" << it->first << "]=" << it->second;
	}

	return (os);
}

template<typename T>
struct TypeTraits<SparseArray<T, NullType> >
{
	typedef SparseArray<T, NullType> & Reference;
	typedef const SparseArray<T, NullType> & ConstReference;
};

template<typename T, typename TL, typename TR, typename TOP>
struct SparseArray<T, BiOp<TL, TR, TOP> >
{
	typedef BiOp<TL, TR, TOP> OpType;
	typedef typename OpType::ValueType ValueType;

	typename OpType::LReference lhs_;
	typename OpType::LReference rhs_;

	typedef typename OpType::const_iterator const_iterator;
	typedef typename OpType::iterator iterator;

	SparseArray(TL const &lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (OpType::op(lhs_, rhs_, s));
	}
	inline const_iterator begin() const
	{
		return (OpType::begin(lhs_, rhs_));
	}

	inline const_iterator end() const
	{
		return (OpType::end(lhs_, rhs_));
	}

};
template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
SparseArray<typename TypeOpTraits<TVL, TVR, arithmetic::OpAddition>::ValueType
		,NullType>  //
operator+(SparseArray<TVL, TLExpr> const & lhs
		, SparseArray<TVR, TRExpr> const & rhs)
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpAddition>::ValueType ValueType;

	SparseArray<ValueType, NullType> v_;
	for (typename SparseArray<TVL, TLExpr>::const_iterator it = lhs.begin();
			it != lhs.end(); ++it)
	{
		v_.insert(std::make_pair(it->first, lhs[it->first]));
	}
	for (typename SparseArray<TVR, TRExpr>::const_iterator it = rhs.begin();
			it != rhs.end(); ++it)
	{
		if (v_.find(it->first) == v_.end())
		{
			v_.insert(std::make_pair(it->first, rhs[it->first]));
		}
		else
		{
			v_[it->first] += rhs[it->first];
		}
	}
	return v_;
}

template<typename TVL, typename TLExpr>
SparseArray<TVL, NullType>  //
operator+(SparseArray<TVL, TLExpr> const & lhs , TVL const & rhs)
{
	typedef TVL ValueType;

	SparseArray<ValueType, NullType> v_;
	for (typename SparseArray<TVL, TLExpr>::const_iterator it = lhs.begin();
			it != lhs.end(); ++it)
	{
		v_.insert(std::make_pair(it->first, lhs[it->first]));
	}
	v_[-1] += rhs;

	return v_;
}
template<typename TVL, typename TRExpr>
SparseArray<TVL, NullType>  //
operator+(TVL const & lhs, SparseArray<TVL, TRExpr> const & rhs)
{
	typedef TVL ValueType;

	SparseArray<ValueType, NullType> v_;
	for (typename SparseArray<TVL, TRExpr>::iterator it = rhs.begin();
			it != rhs.end(); ++it)
	{
		v_.insert(std::make_pair(it->first, lhs[it->first]));
	}
	v_[-1] += lhs;
	return v_;
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpSubtraction>::ValueType
		,NullType>  //
operator-(SparseArray<TVL, TLExpr> const & lhs
		, SparseArray<TVR, TRExpr> const & rhs)
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpSubtraction>::ValueType ValueType;

	SparseArray<ValueType, NullType> v_;

	for (typename SparseArray<TVL, TLExpr>::const_iterator it = lhs.begin();
			it != lhs.end(); ++it)
	{
		v_.insert(std::make_pair(it->first, lhs[it->first]));
	}
	for (typename SparseArray<TVR, TRExpr>::const_iterator it = rhs.begin();
			it != rhs.end(); ++it)
	{
		if (v_.find(it->first) == v_.end())
		{
			v_.insert(std::make_pair(it->first, -rhs[it->first]));
		}
		else
		{
			v_[it->first] -= rhs[it->first];
		}
	}
	return v_;
}
template<typename TVL, typename TLExpr>
SparseArray<TVL, NullType>  //
operator-(SparseArray<TVL, TLExpr> const & lhs , TVL const & rhs)
{
	typedef TVL ValueType;

	SparseArray<ValueType, NullType> v_;
	for (typename SparseArray<TVL, TLExpr>::const_iterator it = lhs.begin();
			it != lhs.end(); ++it)
	{
		v_.insert(std::make_pair(it->first, lhs[it->first]));
	}
	v_[-1] -= rhs;

	return v_;
}
template<typename TV, typename TRExpr>
SparseArray<TV, NullType>  //
operator-(TV const & lhs, SparseArray<TV, TRExpr> const & rhs)
{
	typedef TV ValueType;

	SparseArray<ValueType, NullType> v_;
	for (typename SparseArray<TV, TRExpr>::iterator it = rhs.begin();
			it != rhs.end(); ++it)
	{
		v_.insert(std::make_pair(it->first, -lhs[it->first]));
	}
	v_[-1] += lhs;
	return v_;
}

//--------------------------------------------------------------------------------------------
template<typename TV, typename TL>
struct SparseArray<TV, UniOp<SparseArray<TV, TL> , arithmetic::OpNegative> >
{
	typedef SparseArray<TV, UniOp<SparseArray<TV, TL> , arithmetic::OpNegative> > ThisType;

	typedef TV ValueType;

	typename TypeTraits<SparseArray<TV, TL> >::ConstReference lhs_;

	typedef typename SparseArray<TV, TL>::const_iterator const_iterator;

	SparseArray(SparseArray<TV, TL> const & lhs) :
			lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t const & s) const
	{
		return (-lhs_[s]);
	}

	const_iterator begin() const
	{
		return (lhs_.begin());
	}
	const_iterator end() const
	{
		return (lhs_.end());
	}

};

template<typename TV, typename TL> //
inline SparseArray<TV, UniOp<SparseArray<TV, TL> , arithmetic::OpNegative> >  //
operator -(SparseArray<TV, TL> const & lhs)
{
	return (SparseArray<TV, UniOp<SparseArray<TV, TL> , arithmetic::OpNegative> >(
			lhs));
}
template<typename TVL, typename TLExpr, typename TVR>
struct SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType
		,
		BiOp<SparseArray<TVL, TLExpr> ,TVR, arithmetic::OpMultiplication> >
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	typedef BiOp<SparseArray<TVL, TLExpr> ,TVR , arithmetic::OpMultiplication> OpType;
	typedef SparseArray<ValueType, OpType> ThisType;

	typename TypeTraits<SparseArray<TVL, TLExpr> >::ConstReference lhs_;
	typename TypeTraits<TVR>::ConstReference rhs_;

	typedef typename SparseArray<TVL, TLExpr>::const_iterator const_iterator;

	SparseArray(SparseArray<TVL, TLExpr> const & lhs,TVR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}
	~SparseArray()
	{
	}
	inline ValueType operator[](size_t const & s) const
	{
		return (lhs_[s] * rhs_);
	}

	const_iterator begin() const
	{
		return (lhs_.begin());
	}
	const_iterator end() const
	{
		return (lhs_.end());
	}

};
template<typename TVL, typename TLExpr, typename TVR> //
inline SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType
		,
		BiOp<SparseArray<TVL, TLExpr> ,TVR, arithmetic::OpMultiplication> >  //
operator *(SparseArray<TVL, TLExpr> const & lhs,TVR const & rhs)
{
	typedef SparseArray<
			typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType
			,
			BiOp<SparseArray<TVL, TLExpr> ,TVR , arithmetic::OpMultiplication> > ResultType;
	return (ResultType(lhs, rhs));
}

template<typename TVL, typename TVR, typename TRExpr>
struct SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType
		,
		BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpMultiplication> >
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	typedef BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpMultiplication> OpType;
	typedef SparseArray<ValueType, OpType> ThisType;

	typename TypeTraits<TVL>::ConstReference lhs_;

	typename TypeTraits<SparseArray<TVR, TRExpr> >::ConstReference rhs_;

	typedef typename SparseArray<TVR, TRExpr>::const_iterator const_iterator;

	SparseArray(TVL const & lhs, SparseArray<TVR, TRExpr> const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}
	~SparseArray()
	{
	}
	inline ValueType operator[](size_t const & s) const
	{
		return (lhs_ * rhs_[s]);
	}

	const_iterator begin() const
	{
		return (rhs_.begin());
	}
	const_iterator end() const
	{
		return (rhs_.end());
	}

};
template<typename TVL, typename TVR, typename TRExpr> //
inline SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType
		,
		BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpMultiplication> >  //
operator *(TVL const & lhs, SparseArray<TVR, TRExpr> const & rhs)
{
	typedef SparseArray<
			typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType
			,
			BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpMultiplication> > ResultType;
	return (ResultType(lhs, rhs));
}

//--------------------------------------------------------------------------------------------

template<typename TVL, typename TLExpr, typename TVR>
struct SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ,
		BiOp<SparseArray<TVL, TLExpr> ,TVR, arithmetic::OpDivision> >
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ValueType;
	typedef BiOp<SparseArray<TVL, TLExpr> ,TVR , arithmetic::OpDivision> OpType;
	typedef SparseArray<ValueType, OpType> ThisType;

	typename TypeTraits<SparseArray<TVL, TLExpr> >::ConstReference lhs_;
	typename TypeTraits<TVR>::ConstReference rhs_;

	typedef typename SparseArray<TVL, TLExpr>::const_iterator const_iterator;

	SparseArray(SparseArray<TVL, TLExpr> const & lhs,TVR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}
	~SparseArray()
	{
	}
	inline ValueType operator[](size_t const & s) const
	{
		return (lhs_[s] / rhs_);
	}

	const_iterator begin() const
	{
		return (lhs_.begin());
	}
	const_iterator end() const
	{
		return (lhs_.end());
	}

};
template<typename TVL, typename TLExpr, typename TVR> //
inline SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ,
		BiOp<SparseArray<TVL, TLExpr> ,TVR, arithmetic::OpDivision> >  //
operator /(SparseArray<TVL, TLExpr> const & lhs,TVR const & rhs)
{
	typedef SparseArray<
			typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ,
			BiOp<SparseArray<TVL, TLExpr> ,TVR , arithmetic::OpDivision> > ResultType;
	return (ResultType(lhs, rhs));
}

template<typename TVL, typename TVR, typename TRExpr>
struct SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ,
		BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpDivision> >
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ValueType;
	typedef BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpDivision> OpType;
	typedef SparseArray<ValueType, OpType> ThisType;

	typename TypeTraits<TVL>::Reference lhs_;

	typename TypeTraits<SparseArray<TVR, TRExpr> >::Reference rhs_;

	typedef typename SparseArray<TVR, TRExpr>::const_iterator const_iterator;

	SparseArray(TVL const & lhs, SparseArray<TVR, TRExpr> const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

	~SparseArray()
	{
	}

	inline ValueType operator[](size_t const & s) const
	{
		return (lhs_ / rhs_[s]);
	}

	const_iterator begin() const
	{
		return (rhs_.begin());
	}
	const_iterator end() const
	{
		return (rhs_.end());
	}

};

template<typename TVL, typename TVR, typename TRExpr> //
inline SparseArray<
		typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ,
		BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpDivision> >  //
operator /(TVL const & lhs, SparseArray<TVR, TRExpr> const & rhs)
{
	typedef SparseArray<
			typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ,
			BiOp<TVL, SparseArray<TVR, TRExpr> ,arithmetic::OpDivision> > ResultType;
	return (ResultType(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TR> //
inline typename TypeOpTraits<TVL, typename TR::ValueType, arithmetic::OpDivision>::ValueType //
Dot(SparseArray<TVL, TLExpr> const & lhs,TR const & rhs)
{
	typename TypeOpTraits<TVL, typename TR::ValueType, arithmetic::OpDivision>::ValueType res;
	res = 0;

	for (typename SparseArray<TVL, TLExpr>::iterator it = lhs.begin();
			it != lhs.end(); ++it)
	{
		res += lhs[lhs->first] * rhs[rhs->first];
	}

	return (res);
}

} // namespace simpla
#endif /* SPARSE_ARRAY_H_ */
