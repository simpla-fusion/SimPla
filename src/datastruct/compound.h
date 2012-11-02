/*
 * compound_object.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef COMPOUND_OBJECT_H_
#define COMPOUND_OBJECT_H_

#include "include/simpla_defs.h"
#include <typeinfo>
#include "engine/object.h"

namespace simpla
{

struct CompoundObject: public Object
{
	std::map<std::string, TR1::shared_ptr<Object> > objects;
	ptree properties;

	virtual bool Empty() const
	{
		return objects.empty();
	}
	virtual bool CheckType(std::type_info const &tinfo) const
	{
		return tinfo == typeid(CompoundObject);
	}

	virtual bool CheckValueType(std::type_info const &) const
	{
		return false;
	}

}
;
}  // namespace simpla

#endif /* COMPOUND_OBJECT_H_ */
