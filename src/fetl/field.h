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
#include "engine/object.h"

namespace simpla
{

template<typename TGeometry, typename TValue>
struct Field: public TGeometry,
		public TGeometry::template Container<TValue>,
		public Object
{
public:

	typedef TValue value_type;

	typedef TGeometry geometry_type;

	typedef typename TGeometry::template Container<TValue> container_type;

	typedef Field<geometry_type, value_type> this_type;

	Field()
	{
	}

	template<typename TG, typename TS>
	Field(TG const & g, TS const& e) :
			geometry_type(g), container_type(e)
	{
	}

	template<typename TG, typename TE>
	Field(Field<TG, TE> const & f) :
			geometry_type(f), container_type(f)
	{
	}

	Field(typename geometry_type::Mesh const & g) :
			geometry_type(g),

			container_type(
					(geometry_type::template makeContainer<value_type>()))
	{
	}

	Field(this_type const & f) = default;

	Field(this_type &&rhs) = default;

	virtual ~Field()
	{
	}

	void swap(this_type & rhs)
	{
		geometry_type::swap(rhs);
		container_type::swap(rhs);
	}

	inline bool CheckType(std::type_info const & tinfo) const

	{
		return (tinfo == typeid(this_type));
	}

	inline this_type & operator=(this_type const & rhs)
	{
		geometry_type::mesh->Assign(*this, rhs);
		return (*this);
	}

	template<typename TR> inline this_type &
	operator=(Field<TGeometry, TR> const & rhs)
	{
		geometry_type::mesh->Assign(*this, rhs);
		return (*this);
	}

//	inline auto Get(CoordinatesType const &x,Real effect_radius=0)const
//	DECL_RET_TYPE( (geometry.IntepolateFrom(*this,x,effect_radius)))
//
//	inline auto Put(ValueType const & v,CoordinatesType const &x,Real effect_radius=0)
//	DECL_RET_TYPE(( geometry.IntepolateTo(*this,v,x,effect_radius)))

};
//template<typename T> struct is_Field
//{
//	static const bool value = false;
//};
//
//template<typename TG, typename TE> struct is_Field<Field<TG, TE> >
//{
//	static const bool value = true;
//};
//

template<typename TG, typename TL, typename TR> typename TG::Mesh const * //
get_mesh(Field<TG, TL> const & l, TR const & r)
{
	return (l.mesh);
}

template<typename TG, typename TL, int IR, typename TR> typename TG::Mesh const * //
get_mesh(TL const & l, Field<TG, TR> const & r)
{
	return (r.mesh);
}

template<typename TGL, typename TL, typename TGR, typename TR> typename TGL::Mesh const * //
get_mesh(Field<TGL, TL> const & l, Field<TGR, TR> const & r)
{
	return (l.mesh);
}

}
// namespace simpla

#endif /* FIELD_H_ */
