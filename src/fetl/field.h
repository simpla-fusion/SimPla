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

	Field(typename GeometryType::Mesh const & g) :
			GeometryType(g), BaseType(TGeometry(g).get_num_of_elements())
	{
	}

	Field(ThisType const &) = delete;

	Field(ThisType &&rhs) :
			GeometryType(rhs), BaseType(rhs)
	{
	}

	virtual ~Field()
	{
	}

	void swap(ThisType & rhs)
	{
		GeometryType::swap(rhs);
		BaseType::swap(rhs);
	}

	inline bool CheckType(std::type_info const & tinfo)
	{
		return (tinfo == typeid(ThisType));
	}

	inline ThisType & operator=(ThisType const & rhs)
	{
		GeometryType::mesh->Assign(*this, rhs);
		return (*this);
	}

	template<typename TR> inline ThisType &
	operator=(Field<TGeometry, TR> const & rhs)
	{
		GeometryType::mesh->Assign(*this, rhs);
		return (*this);
	}

//	inline auto Get(CoordinatesType const &x,Real effect_radius=0)const
//	DECL_RET_TYPE( (geometry.IntepolateFrom(*this,x,effect_radius)))
//
//	inline auto Put(ValueType const & v,CoordinatesType const &x,Real effect_radius=0)
//	DECL_RET_TYPE(( geometry.IntepolateTo(*this,v,x,effect_radius)))

};
template<typename T> struct is_Field
{
	static const bool value = false;
};

template<typename TG, typename TE> struct is_Field<Field<TG, TE> >
{
	static const bool value = true;
};

template<typename TG, typename T>
struct is_storage_type<Field<TG, T> >
{
	static const bool value = is_storage_type<T>::value;
};

template<typename TM, int IL, typename TL, typename TR> TM const * //
get_mesh(Field<Geometry<TM, IL>, TL> const & l, TR const & r)
{
	return (l.mesh);
}

template<typename TM, typename TL, int IR, typename TR> TM const * //
get_mesh(TL const & l, Field<Geometry<TM, IR>, TR> const & r)
{
	return (r.mesh);
}

template<typename TM, int IL, typename TL, int IR, typename TR> TM const * //
get_mesh(Field<Geometry<TM, IL>, TL> const & l,
		Field<Geometry<TM, IR>, TR> const & r)
{
	return (l.mesh);
}

template<typename T, typename TR> struct ColneField;

template<typename TG, typename TE, typename TR>
struct ColneField<Field<TG, TE>, TR>
{
	typedef Field<TG, TR> type;
};
}
// namespace simpla

#endif /* FIELD_H_ */
