/*
 * object.h
 *
 *  Created on: 2014年11月18日
 *      Author: salmon
 */

#ifndef CORE_PHYSICS_PHYSICAL_OBJECT_H_
#define CORE_PHYSICS_PHYSICAL_OBJECT_H_

#include <iostream>

namespace simpla
{
class DataSet;
class Properties;

struct PhysicalObject
{
	//! Default constructor
	PhysicalObject() :
			is_valid_(false)
	{
	}
	//! destroy.
	virtual ~PhysicalObject()
	{
	}

//	PhysicalObject(const PhysicalObject&); // copy constructor.
//	PhysicalObject(PhysicalObject &&); // move constructor.

//	virtual Properties const & properties(
//			std::string const & name = "") const=0;
//
//	virtual Properties & properties(std::string const & name = "") =0;

	virtual std::string get_type_as_string() const=0;

	virtual DataSet dataset() const =0; //!< return the data set of PhysicalObject

	bool is_valid()
	{
		return is_valid_;
	}

	virtual bool initialize()
	{
		is_valid_ = true;
		return true;
	}
	virtual void synchronize()
	{

	}
	virtual void asynchronize()
	{

	}
	virtual bool update()
	{
		is_valid_ = true;
		return true;
	}
private:

	bool is_valid_ = false;

};
}  // namespace simpla

#endif /* CORE_PHYSICS_PHYSICAL_OBJECT_H_ */
