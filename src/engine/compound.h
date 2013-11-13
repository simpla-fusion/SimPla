/*
 * compound_object.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef COMPOUND_OBJECT_H_
#define COMPOUND_OBJECT_H_

#include "object.h"
#include <typeinfo>
#include <map>
#include <utility>
namespace simpla
{
class CompoundObject: public Object, public std::map<std::string, Object>
{
public:

	typedef std::map<std::string, Object> base_type;
	typedef CompoundObject this_type;
	CompoundObject() :
			Object(typeid(this_type))
	{
	}

	CompoundObject(CompoundObject const &r) :
			Object(r)
	{
	}

	~CompoundObject()
	{
	}

	virtual void swap(CompoundObject & rhs)
	{

		Object::swap(rhs);
		base_type::swap(rhs);
	}
}
;

}
// namespace simpla

#endif /* COMPOUND_OBJECT_H_ */
