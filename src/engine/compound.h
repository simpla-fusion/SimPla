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

	std::map<std::string, std::shared_ptr<Object> > childs;

public:

	CompoundObject();

	~CompoundObject();

	virtual void swap(CompoundObject & rhs)
	{
		childs.swap(rhs.childs);
		Object::swap(rhs);
	}

	virtual bool IsEmpty() const
	{
		return (childs.empty());
	}
	virtual bool CheckType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(CompoundObject));
	}



	boost::optional<std::shared_ptr<Object> > Find(std::string const &name);

	boost::optional<std::shared_ptr<const Object> > Find(
			std::string const &name) const;


	inline boost::optional<Object> operator[](std::string const &name)
	{
		return (*Find(name));
	}

	inline boost::optional<const Object> operator[](
			std::string const &name) const
	{
		return (*Find(name));
	}

	template<typename T>
	inline boost::optional<T> get(std::string const & key)
	{
		boost::optional<std::shared_ptr<Object> > res = Find(key);

		return boost::optional<T>(!(!res), (*res)->as<T>());
	}

	template<typename T>
	inline boost::optional<const T> get(std::string const & key) const
	{
		boost::optional<const Object> res = Find(key);

		return boost::optional<const T>(!(!res), res->as<const T>());
	}



	void Add(std::string const & name, std::shared_ptr<Object> obj);

	void Delete(std::string const & name);

}
;

}
// namespace simpla

#endif /* COMPOUND_OBJECT_H_ */
