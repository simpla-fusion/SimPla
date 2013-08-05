/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_
#include <memory> // for shared_ptr
#include <algorithm> //for swap
#include <utility>
#include <type_traits>

namespace simpla
{

template<typename TGeometry, typename TStorage>
struct Field: public TGeometry, public TStorage
{
public:
	typedef TGeometry GeometryType;

	typedef TStorage BaseType;

	typedef Field<GeometryType, TStorage> ThisType;

	typedef typename TGeometry::CoordinatesType CoordinatesType;

	Field()
	{
	}

	template<typename TG, typename TE>
	Field(TG const & g, TE e) :
			GeometryType(g), BaseType(e)
	{
	}

	template<typename TG, typename TE>
	Field(Field<TG, TE> const & f) :
			GeometryType(f), BaseType(f)
	{
	}

	Field(typename GeometryType::Grid const & g) :
			GeometryType(g), BaseType(TGeometry(g).get_num_of_elements())
	{
	}

	Field(ThisType const &) = delete;

	Field(ThisType &&rhs) :
			GeometryType(rhs), BaseType(rhs)
	{
	}

	void swap(ThisType & rhs)
	{
		GeometryType::swap(rhs);
		BaseType::swap(rhs);
	}

	virtual ~Field()
	{
	}

	inline ThisType & operator=(ThisType const & rhs)
	{
		GeometryType::grid->Assign(*this, rhs);
		return (*this);
	}

	template<typename TR>
	inline typename std::enable_if<is_Field<TR>::value, ThisType &>::type //
	operator=(TR const & rhs)
	{
		GeometryType::grid->Assign(*this, rhs);
		return (*this);
	}

//	inline auto Get(CoordinatesType const &x,Real effect_radius=0)const
//	DECL_RET_TYPE( (geometry.IntepolateFrom(*this,x,effect_radius)))
//
//	inline auto Put(ValueType const & v,CoordinatesType const &x,Real effect_radius=0)
//	DECL_RET_TYPE(( geometry.IntepolateTo(*this,v,x,effect_radius)))

};

template<typename TG, typename T>
struct is_storage_type<Field<TG, T> >
{
	static const bool value = is_storage_type<T>::value;
};

template<typename TL, typename TR> auto get_grid(TL const & l,
		TR const & r)
		-> typename std::enable_if<is_Field<TL>::value,typename TL::Geometry::Grid >::type const *
{
	return (l.grid);
}

template<typename TL, typename TR> auto get_grid(TL const & l, TR const & r)
-> typename std::enable_if<(!is_Field<TL>::value) && is_Field<TR>::value,
typename TR::Geometry::Grid>::type const *
{
	return (r.grid);
}

}
// namespace simpla

#endif /* FIELD_H_ */
