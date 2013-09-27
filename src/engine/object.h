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
#include "utilities/memory_pool.h"
#include "utilities/properties.h"
namespace simpla
{
class Object
{
public:

	PTree properties;

	Object() :
			data_(nullptr), size_in_bytes_(0)
	{
	}
	Object(size_t size_in_byte) :
			data_(MemoryPool::instance().alloc(size_in_byte)), size_in_bytes_(
					size_in_byte)
	{
	}
	Object(std::shared_ptr<ByteType> d, size_t s = 0) :
			data_(d), size_in_bytes_(s)
	{
	}

	Object(Object const &) = default;

	Object(Object && rhs) :
			size_in_bytes_(rhs.size_in_bytes_), properties(rhs.properties)
	{
		rhs.data_.swap(rhs.data_);
	}

	inline virtual ~Object()
	{
		MEMPOOL.release();
	}
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
	T const & as()const
	{
		if(!CheckType(typeid(T)))
		{
			ERROR<<"Can not convert to type "<<typeid(T).name();
		}
		return (*dynamic_cast<const T>(data_));
	}

	virtual void swap(Object & rhs)
	{
		properties.swap(rhs.properties);

		data_.swap(rhs.data_);
		std::swap(size_in_bytes_,rhs.size_in_bytes_);

	}

	// Metadata ------------------------------------------------------------

	virtual bool CheckType(std::type_info const &) const=0;

	bool IsEmpty() const
	{
		return (data_.get()==nullptr);
	}

protected:
	std::shared_ptr<ByteType> data_;
	size_t size_in_bytes_;

};

}
 // namespace simpla

#endif /* OBJECT_H_ */
