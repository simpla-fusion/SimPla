/*
 * compound_object.h
 *
 *  Created on: 2012-11-1
 *      Author: salmon
 */

#ifndef COMPOUND_OBJECT_H_
#define COMPOUND_OBJECT_H_

#include "include/simpla_defs.h"
#include "object.h"

#include <typeinfo>

namespace simpla
{
class CompoundObject: public Object
{
public:
	std::map<std::string, TR1::shared_ptr<Object> > childs;

	CompoundObject();

	~CompoundObject();

	virtual void swap(CompoundObject & rhs)
	{
		childs.swap(rhs.childs);
		Object::swap(rhs);
	}

	static TR1::shared_ptr<CompoundObject> Create(BaseContext * ctx,
			ptree const & pt);

	virtual bool IsEmpty() const
	{
		return (childs.empty());
	}
	virtual bool CheckType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(CompoundObject));
	}

	TR1::shared_ptr<Object> operator[](std::string const &name);

	TR1::shared_ptr<Object> operator[](std::string const &name) const;

	bool CheckObjectType(std::string const & name,
			std::type_info const &) const;

	boost::optional<TR1::shared_ptr<Object> > FindObject(
			std::string const & name);

	boost::optional<TR1::shared_ptr<const Object> > FindObject(
			std::string const & name) const;

	void DeleteObject(std::string const & name);

}
;

}
// namespace simpla

#endif /* COMPOUND_OBJECT_H_ */
