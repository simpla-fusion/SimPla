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
#include "datastruct/array.h"
namespace simpla
{

template<typename TGeometry, typename TV>
struct Field: public Array<TV>
{
public:

	typedef Array<TV> BaseType;
	typedef Field<TGeometry, TV> ThisType;
	typedef TGeometry Grid;
	typedef typename Grid::Coordinates Coordinates;

	std::shared_ptr<const Grid> grid;

	Field()
	{
	}

	Field(std::shared_ptr<Grid> g, size_t value_size = sizeof(Value)) :
			BaseType(grid->get_num_of_elements(), value_size), grid(g)
	{
	}

	Field(ThisType const &) = delete;

	void swap(ThisType & rhs)
	{
		BaseType::swap(rhs);
		std::swap(grid, rhs.grid);
	}

	virtual ~Field()
	{
	}

	inline Value Get(Coordinates const &x,Real effect_radius=0)const
	{
		return (grid->IntepolateFrom(*this,x,effect_radius));
	}

	inline void Put(Value const & v,Coordinates const &x,Real effect_radius=0)
	{
		grid->IntepolateTo(*this,v,x,effect_radius);
	}

//	bool CheckType(std::type_info const &rhs) const
//	{
//		return (typeid(ThisType) == rhs || BaseType::CheckType(rhs));
//	}

// Assignment --------

};

}  // namespace simpla

#endif /* FIELD_H_ */
