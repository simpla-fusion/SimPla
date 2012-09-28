/*
 * sparse_matrix.h
 *
 *  Created on: 2012-3-26
 *      Author: salmon
 */

#ifndef SPARSE_MATRIX_H_
#define SPARSE_MATRIX_H_
#include <vector>
#include <map>
#include <iostream>
#include "include/simpla_defs.h"
#include "engine/object.h"
#include "primitives/operation.h"
#include "primitives/sparse_array.h"
#include "primitives/typetraits.h"
namespace simpla
{

template<typename T, typename TExpr = NullType> struct SparseMatrix;

template<typename T>
class SparseMatrix<T, NullType> : //
public std::map<size_t, SparseArray<T, NullType> >
{

public:
	typedef T ValueType;
	typedef SparseArray<T, NullType> VectorType;
	typedef SparseMatrix<T, NullType> ThisType;
	typedef std::map<size_t, SparseArray<T, NullType> > BaseType;

	SparseMatrix()
	{
	}
	~SparseMatrix()
	{
	}

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
		for (auto it = rhs.begin(); it != rhs.end(); ++it)
		{
			t.insert(std::make_pair(it->first, rhs[it->first]));
		}
		t.swap(*this);
		return (*this);
	}

	static const VectorType ZERO;
};
template<typename T> typename SparseMatrix<T, NullType>::ValueType const //
SparseMatrix<T, NullType>::ZERO;

template<typename T>
struct TypeTraits<SparseMatrix<T, NullType> >
{
	typedef SparseMatrix<T, NullType> & Reference;
	typedef const SparseMatrix<T, NullType> & ConstReference;
};

template<typename T, typename TExpr> std::ostream &
operator<<(std::ostream& os, const SparseMatrix<T, TExpr> & tv)
{
	for (auto it = tv.begin(); it != tv.end(); ++it)
	{
		for (auto jt = it->second.begin(); jt != it->second.end(); ++jt)
		{
			os << "\t[" << it->first << "," << jt->first << "]=" << jt->second;
		}
		os << std::endl;
	}
	return (os);
}
} // namespace simpla

#endif /* SPARSE_MATRIX_H_ */
