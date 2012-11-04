/*
 * object.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef OBJECT_H_
#define OBJECT_H_
#include <typeinfo>
#include "utilities/properties.h"
namespace simpla
{
class Object
{
public:
	Object()
	{
	}
	inline virtual ~Object()
	{
	}

	ptree properties;

	virtual void swap(Object & rhs)
	{
		properties.swap(rhs.properties);
	}

	// Metadata ------------------------------------------------------------

	virtual bool CheckType(std::type_info const &) const=0;

	virtual bool CheckValueType(std::type_info const &) const=0;

	virtual bool Empty() const=0;

};

inline bool Object::CheckValueType(std::type_info const &) const

{
	return false;
}
}  // namespace simpla

#endif /* OBJECT_H_ */
