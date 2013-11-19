/*
 * object.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef OBJECT_H_
#define OBJECT_H_
#include "utilities/log.h"
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

namespace simpla
{
class Object
{
public:

	typedef std::multimap<std::string, Object> children_type

	Object() :
			data_(nullptr), type_id_(std::type_index(typeid(void)))
	{
	}

	Object(std::type_info const & tinfo) :
			data_(nullptr), type_id_(std::type_index(tinfo))
	{
	}

	Object(Object const & r) :
			data_(r.data_), type_id_(r.type_id_)
	{
	}

	Object(Object &&r) :
			data_(r.data_), type_id_(r.type_id_)
	{
	}

	template<typename T>
	Object(std::shared_ptr<T> d) :
			data_(std::static_pointer_cast<void>(d)), type_id_(
					std::type_index(typeid(T)))
	{
	}

	template<typename T>
	Object(T* d) :
			data_(std::static_pointer_cast<void>(std::shared_ptr<T>(d))), type_id_(
					std::type_index(typeid(T)))
	{
	}

	virtual ~Object()
	{
	}

	template<typename T>
	std::shared_ptr<T> as()
	{
		if (!CheckType<T>())
		{
			ERROR << "This is not a " << typeid(T).name() << " !";
		}
		return (std::static_pointer_cast<T>(data_));
	}
	template<typename T>
	std::shared_ptr<const T> as() const
	{
		if (!CheckType<T>())
		{
			ERROR << "Can not convert to type " << typeid(T).name();
		}
		return (std::static_pointer_cast<const T>(data_));
	}

	virtual void swap(Object & rhs)
	{

		data_.swap(rhs.data_);

	}

// Metadata ------------------------------------------------------------

	template<typename T>
	inline bool CheckType() const
	{
		return std::type_index(typeid(T)) == type_id_;
	}

	bool IsEmpty() const
	{
		return (data_.get() == nullptr);
	}

	inline children_type & GetChildren()
	{
		return children_;
	}

	inline children_type const & GetChildren() const
	{
		return children_;
	}

	std::shared_ptr<void> data_;
	std::type_index type_id_;
	std::string name_;
	children_type children_;
};

template<typename T, typename ... Args>
inline Object CreateObjec(Args &... args)
{
	return Object(new T(std::forward<Args>(args)...));
}

}
template<typename T> class ObjectWrapper;
// namespace simpla

#endif /* OBJECT_H_ */
