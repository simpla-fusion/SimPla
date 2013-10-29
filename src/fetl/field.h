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

/***
 *
 * @brief Field
 *
 * @ingroup Field Expression
 *
 */

template<typename TGeometry, typename TValue>
struct Field: public TGeometry, public TGeometry::template Container<TValue>

{
public:

	typedef TGeometry geometry_type;

	typedef typename TGeometry::template Container<TValue> container_type;

	typedef typename geometry_type::index_type index_type;

	typedef typename container_type::value_type value_type;

	typedef Field<geometry_type, value_type> this_type;

	Field(typename geometry_type::Mesh const & g) :
			geometry_type(g),

			container_type(
					std::move(
							geometry_type::template MakeContainer<value_type>()))
	{
	}

	Field() = default;

	Field(this_type const & f) = delete;

	Field(this_type &&rhs) = delete;

	virtual ~Field()
	{
	}

	void swap(this_type & rhs)
	{
		geometry_type::swap(rhs);
		container_type::swap(rhs);
	}

	template<typename Fun> inline void ForEach(Fun const & fun)
	{
		geometry_type::ForEach(*this, fun);
	}

	inline this_type & operator=(this_type const & rhs)
	{
		geometry_type::mesh->ForEach("Center",

		[this, &rhs](typename geometry_type::index_type s)
		{
			(*this)[s]=rhs[s];
		}

		);
		return (*this);
	}

#define DECL_SELF_ASSIGN( _OP_ )                                                  \
	template<typename TR> inline this_type &                                      \
	operator _OP_(Field<TGeometry, TR> const & rhs)                               \
	{                                                                             \
		geometry_type::ForEach([this, &rhs](typename geometry_type::index_type s)         \
		{	(*this)[s] _OP_ rhs[s];});                                            \
		return (*this);                                                           \
	}

	DECL_SELF_ASSIGN(=)DECL_SELF_ASSIGN(+=)
	DECL_SELF_ASSIGN(-=)
	DECL_SELF_ASSIGN(*=)
	DECL_SELF_ASSIGN(/=)
#undef DECL_SELF_ASSIGN
//	inline auto Get(CoordinatesType const &x,Real effect_radius=0)const
//	DECL_RET_TYPE( (geometry.IntepolateFrom(*this,x,effect_radius)))
//
//	inline auto Put(ValueType const & v,CoordinatesType const &x,Real effect_radius=0)
//	DECL_RET_TYPE(( geometry.IntepolateTo(*this,v,x,effect_radius)))

};
}
 // namespace simpla

#endif /* FIELD_H_ */
