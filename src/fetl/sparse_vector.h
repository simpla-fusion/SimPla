/*
 * sparse_vector.h
 *
 *  Created on: 2013-7-11
 *      Author: salmon
 */

#ifndef SPARSE_VECTOR
#define SPARSE_VECTOR

#include <map>
#include <utility>
#include <map>

namespace simpla
{

template<typename TG, typename TE> struct Field;

template<typename TV> struct SparseMatrix;
template<typename T> class SparseVector;
template<typename TV> struct PlaceHolder;

template<typename TV>
struct SparseMatrix<std::map<std::pair<size_t, size_t>, TV> > : //
public std::map<std::pair<size_t, size_t>, TV>
{

	typedef std::map<std::pair<size_t, size_t>, TV> MatrixType;

	typedef std::pair<size_t, size_t> IndexType;

	typedef TV ValueType;

	static const size_t MAX_IDX = std::numeric_limits<size_t>::max;

	SparseMatrix()
	{
	}

	SparseVector<ThisType> & operator[](size_t col_idx)
	{
		return (SparseVector<ThisType>(this, col_idx));
	}

	ThisType & operator=(Field<TG, TE> const &f)
	{
		size_t num = f.get_num_of_elements();
		for (size_t s = 0; s < num; ++s)
		{
			this->operator[](s) = f[s];
		}
		return (*this);
	}

};

template<typename TV>
class SparseVector<std::map<std::pair<size_t, size_t>, TV> >
{
	typedef std::map<std::pair<size_t, size_t>, TV> MatrixType;

	MatrixType * m_;
	const size_t col_idx_;

	SparseVector(MatrixType * m, size_t col_idx) :
			m_(m), col_idx_(col_idx)
	{

	}
	inline TV & operator[](size_t row_idx)
	{
		std::pair<size_t, size_t> idx(col_idx_, row_idx);

		auto it = m_->find(idx);

		if (it == m_->end())
		{
			it = m_->insert(std::make_pair(idx, 0));
		}
		return (*it);
	}

	template<typename TR>
	inline SparseVector & operator=(PlaceHolder<TR> const & r)
	{
		r.assign(*this, 1);
	}
	template<typename TR>
	inline SparseVector & operator=(TR const & r)
	{
		PlaceHolder<TR>(std::numeric_limits<size_t>::max, r).assign(*this, 1);
	}
};

template<typename TGeometry, typename TR>
struct Field<TGeometry, PlaceHolder<TR> > : public TGeometry
{
public:
	typedef TGeometry GeometryType;

	typedef Field<GeometryType, PlaceHolder<TR> > ThisType;

	typedef typename GeometryType::CoordinatesType CoordinatesType;

	Field()
	{
	}

	template<typename TG>
	Field(TG const & g) :
			GeometryType(g)
	{
	}

	Field(typename GeometryType::Grid const & g) :
			GeometryType(g)
	{
	}

	Field(ThisType const &) = delete;

	virtual ~Field()
	{
	}

	PlaceHolder<TR> operator[](size_t s)
	{
		return (PlaceHolder<TR>(s,1));
	}

};

template<typename TV> struct PlaceHolder
{
	size_t idx;
	TV value;
	template<typename TVec, typename TR>
	bool assign(TVec * vec, TR const &a)
	{
		(*vec)[idx] += value * a;
	}
};

template<typename > struct is_PlaceHolder
{
	static const bool value = false;
};

template<typename T> struct is_PlaceHolder<PlaceHolder<T> >
{
	static const bool value = true;
};

template<typename TL, typename TR> struct OpMultipliesPlaceHolder;
template<typename TL, typename TR> struct PlaceHolder<
		OpMultipliesPlaceHolder<PlaceHolder<TL>, TR> >
{
	typename ConstReferenceTraits<PlaceHolder<TL>>::type l_;
	typename ConstReferenceTraits<TR>::type r_;
	PlaceHolder(PlaceHolder<TL> const & l, TR const & r) :
			l_(l), r_(r)
	{
	}
	template<typename TV, typename TR>
	inline void assign(TV * v, TR const & a)
	{
		l_.assign(v, a * r_);
	}
};

template<typename TL, typename TR> struct PlaceHolder<
		OpMultipliesPlaceHolder<TL, PlaceHolder<TR>> >
{
	typename ConstReferenceTraits<PlaceHolder<TR>>::type r_;
	typename ConstReferenceTraits<TL>::type l_;
	PlaceHolder(TL const & l, PlaceHolder<TR> const & r) :
			l_(l), r_(r)
	{
	}
	template<typename TV, typename TR>
	inline void assign(TV * v, TR const & a)
	{
		r_.assign(v, a * l_);
	}
};

template<typename TL, typename TR> struct OpPlusPlaceHolder;
template<typename TL, typename TR>
struct PlaceHolder<OpPlusPlaceHolder<PlaceHolder<TL>, PlaceHolder<TR> > >
{
	typename ConstReferenceTraits<PlaceHolder<TL>>::type l_;
	typename ConstReferenceTraits<PlaceHolder<TR>>::type r_;

	PlaceHolder(PlaceHolder<TL> const & l, PlaceHolder<TR> const & r) :
			l_(l), r_(r)
	{
	}
	template<typename TV, typename TR>
	inline void assign(TV * m, TR const & a)
	{
		l_.assign(m, a);
		r_.assign(m, a);
	}
};

template<typename TL, typename TR> struct PlaceHolder<
		OpPlusPlaceHolder<PlaceHolder<TL>, TR> >
{
	typename ConstReferenceTraits<PlaceHolder<TL>>::type l_;
	typename ConstReferenceTraits<TR>::type r_;
	PlaceHolder(PlaceHolder<TL> const & l, TR const & r) :
			l_(l), r_(r)
	{
	}
	template<typename TV, typename TR>
	inline void assign(TV * v, TR const & a)
	{
		l_.assign(v, a);
		PlaceHolder<TR>(std::numeric_limits<size_t>::max, r_).assign(v, a);
	}
};

template<typename TL, typename TR> struct PlaceHolder<
		OpPlusPlaceHolder<TL, PlaceHolder<TR>> >
{
	typename ConstReferenceTraits<PlaceHolder<TR>>::type r_;
	typename ConstReferenceTraits<TL>::type l_;
	PlaceHolder(TL const & l, PlaceHolder<TR> const & r) :
			l_(l), r_(r)
	{
	}
	template<typename TV, typename TR>
	inline void assign(TV * v, TR const & a)
	{
		r_.assign(v, a);
		PlaceHolder<TR>(std::numeric_limits<size_t>::max, l_).assign(v, a);

	}
};

template<typename TL, typename TR>
auto Multiplies(PlaceHolder<TL> const & l, TR const & r)
ENABLE_IF_DECL_RET_TYPE((!is_PlaceHolder<TR>::value),
		( PlaceHolder<OpMultipliesPlaceHolder<PlaceHolder<TL>, TR > >(l,r)))

template<typename TL, typename TR>
auto Multiplies(TL const & r,
		PlaceHolder<TR> const & l)
ENABLE_IF_DECL_RET_TYPE((!is_PlaceHolder<TL>::value),
		( PlaceHolder<OpMultipliesPlaceHolder<TL,PlaceHolder<TR> > >(l,r)))

template<typename TL, typename TR>
auto Divides(PlaceHolder<TL> const & l,
		TR const & r)
ENABLE_IF_DECL_RET_TYPE((!is_PlaceHolder<TR>::value),
		( PlaceHolder<OpMultipliesPlaceHolder<PlaceHolder<TL>, decltype(1.0/r) > >(l,1.0/r)))

template<typename TL>
auto Negate(PlaceHolder<TL> const & l)
DECL_RET_TYPE(( PlaceHolder<OpMultipliesPlaceHolder<TL,double> >(l,-1.0)))

template<typename TL, typename TR>
auto Plus(PlaceHolder<TL> const & l,
		PlaceHolder<TR> const & r)
DECL_RET_TYPE(( PlaceHolder<OpPlusPlaceHolder<PlaceHolder<TL>, PlaceHolder<TR> > >(l,r)))

template<typename TL, typename TR>
auto Plus(TL const & l, PlaceHolder<TR> const & r)
ENABLE_IF_DECL_RET_TYPE((!is_PlaceHolder<TL>::value),
		( PlaceHolder<OpPlusPlaceHolder<TL, PlaceHolder<TR> > >(l,r)))

template<typename TL, typename TR>
auto Plus(PlaceHolder<TL> const & l, TR const & r)
ENABLE_IF_DECL_RET_TYPE((!is_PlaceHolder<TR>::value),
		( PlaceHolder<OpPlusPlaceHolder<PlaceHolder<TL>, TR > >(l,r)))

template<typename TL, typename TR>
auto Minus(PlaceHolder<TL> const & l, PlaceHolder<TL> const & r)
DECL_RET_TYPE((Plus(l,Negate(r))))

}
// namespace simpla

#endif /* SPARSE_VECTOR */
