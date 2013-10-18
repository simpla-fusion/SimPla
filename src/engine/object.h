/*
 * object.h
 *
 *  Created on: 2012-10-31
 *      Author: salmon
 */

#ifndef OBJECT_H_
#define OBJECT_H_
#include <typeinfo>
#include <memory>
#include <algorithm>
#include <string>
#include "utilities/properties.h"
namespace simpla
{
class Object
{
public:

	Object() :
			data_(nullptr)
	{
	}

	Object(std::shared_ptr<ByteType> d) :
			data_(d)
	{
	}

	template<typename T>
	Object(T* d) :
			data_(std::shared_ptr<ByteType>(d))
	{
	}
	Object(Object const &) = default;

	Object(Object && rhs) = default;

	virtual ~Object()
	{
	}

//	template<typename T> inline T & as()
//	{
//		if(!CheckType(typeid(T)))
//		{
//			ERROR<<"Can not convert to type "<<typeid(T).name();
//		}
//		return (*dynamic_cast<T>(data_));
//	}
//	template<typename T>inline
//	T const & as()const
//	{
//		if(!CheckType(typeid(T)))
//		{
//			ERROR<<"Can not convert to type "<<typeid(T).name();
//		}
//		return (*dynamic_cast<const T>(data_));
//	}
	template<typename T>
	T & as()
	{
		if(!CheckType(typeid(T)))
		{
			ERROR<<"Can not convert to type "<<typeid(T).name();
		}
		return (*dynamic_cast<T>(data_));
	}
	template<typename T>
	T const & as() const
	{
		if (!CheckType(typeid(T)))
		{
			ERROR << "Can not convert to type " << typeid(T).name();
		}
		return (*dynamic_cast<const T>(data_));
	}

	virtual void swap(Object & rhs)
	{
		properties.swap(rhs.properties);

		data_.swap(rhs.data_);

	}

// Metadata ------------------------------------------------------------

	virtual bool CheckType(std::type_info const &) const=0;

	bool IsEmpty() const
	{

		return (data_.get() == nullptr);

	}

public:
	PTree properties;
protected:
	std::shared_ptr<ByteType> data_;
	size_t size_in_bytes_;

};

}
// namespace simpla

#endif /* OBJECT_H_ */
