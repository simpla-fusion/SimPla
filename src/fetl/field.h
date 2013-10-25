/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

namespace simpla
{

template<typename TGeometry, typename TValue>
struct Field: public TGeometry, public TGeometry::template Container<TValue>

{
public:

	typedef TValue value_type;

	typedef TGeometry geometry_type;

	typedef typename TGeometry::template Container<TValue> container_type;

	typedef Field<geometry_type, value_type> this_type;

	Field(typename geometry_type::Mesh const & g) :
			geometry_type(g), container_type(
					std::move(
							geometry_type::template makeContainer<value_type>()))
	{
	}

	Field() = default;

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

//	inline this_type & operator=(this_type const & rhs)
//	{
//		geometry_type::mesh->Assign(*this, rhs);
//		return (*this);
//	}

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
} // namespace simpla

#endif /* FIELD_H_ */
