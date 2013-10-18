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
#include <map>
#include <utility>
namespace simpla
{
class CompoundObject: public Object
{

	std::map<std::string, std::shared_ptr<Object> > childs;

public:

	CompoundObject() = default;

	CompoundObject(CompoundObject const &) = default;

	~CompoundObject() = default;

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

<<<<<<< HEAD
	inline boost::optional<std::shared_ptr<Object> > Find(
			std::string const & name)
	{
		auto it = childs.find(name);

		if (it != childs.end())
		{
			return (boost::none);
		}
		else
		{
			return boost::optional<std::shared_ptr<Object> >(it->second);
		}

	}

	inline boost::optional<const std::shared_ptr<Object> > operator[](
			std::string const &name) const
	{
		return (*Find(name));
	}

	inline boost::optional<const std::shared_ptr<Object> > Find(
			std::string const & name) const
	{

		auto it = childs.find(name);

		if (it != childs.end())
		{
			return (boost::none);
		}
		else
		{
			return boost::optional<const std::shared_ptr<Object> >(it->second);
		}
	}

	inline boost::optional<std::shared_ptr<Object> > operator[](
			std::string const &name)
=======


	boost::optional<std::shared_ptr<Object> > Find(std::string const &name);

	boost::optional<std::shared_ptr<const Object> > Find(
			std::string const &name) const;


	inline boost::optional<Object> operator[](std::string const &name)
	{
		return (*Find(name));
	}

	inline boost::optional<const Object> operator[](
			std::string const &name) const
>>>>>>> ddb1baf4864f73bec4047c704d79f5c9a1152544
	{
		return (*Find(name));
	}

	template<typename T>
<<<<<<< HEAD
	inline boost::optional<std::shared_ptr<T>> Get(std::string const & key)
	{
		auto res = Find(key);
		if (!res || !((*res)->CheckType(typeid(T))))
		{
			return boost::none;
		}
		else
		{
			return boost::optional<const std::shared_ptr<T> >(
					dynamic_cast<T>(*res));
		}
	}

	template<typename T>
	inline boost::optional<const std::shared_ptr<T>> Get(
			std::string const & key) const
	{
		auto res = Find(key);
		if (!res || !((*res)->CheckType(typeid(T))))
		{
			return boost::none;
		}
		else
		{
			return boost::optional<const std::shared_ptr<T> >(
					dynamic_cast<const T>(*res));
		}
	}

	inline void Add(std::string const & name, std::shared_ptr<Object> obj)
	{
		childs.insert(std::make_pair(name, obj));
	}

	inline void Delete(std::string const & name)
	{
		auto it = childs.find(name);
		if (it != childs.end())
		{
			childs.erase(it);
		}
	}
=======
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
>>>>>>> ddb1baf4864f73bec4047c704d79f5c9a1152544

}
;

}
// namespace simpla

#endif /* COMPOUND_OBJECT_H_ */
