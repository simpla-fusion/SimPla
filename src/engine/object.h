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
namespace simpla
{
class Object
{
public:
	Object()
	{
	}
	Object(size_t size_in_byte, std::string name = "unnamed") :
			data_(MemoryPool::instance().alloc(size_in_byte)), name_(name)
	{
	}

	inline virtual ~Object()
	{
		MEMPOOL.release();
	}

//	ptree properties;

	virtual void swap(Object & rhs)
	{
		std::swap(data_, rhs.data_);
		std::swap(name_, rhs.name_);
//		properties.swap(rhs.properties);
	}

	// Metadata ------------------------------------------------------------

//	virtual bool CheckType(std::type_info const &) const=0;

	virtual bool IsEmpty() const=0;

protected:
	std::string name_;
	std::shared_ptr<ByteType> data_;

};

}  // namespace simpla

#endif /* OBJECT_H_ */
