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

/** @ingroup physics
 *  @addtogroup physical_object Physical object
 *  @{
 *  ## Summary
 *   - Particle distribution function is a @ref physical_object;
 *   - Electric field is a @ref physical_object
 *   - Magnetic field is a @ref physical_object;
 *   - Plasma density field is a @ref physical_object;
 *   - @ref physical_object is a manifold defined on a domain in configuration space;
 *   - @ref physical_object has properties;
 *   - @ref physical_object can be saved or loaded as DataSet;
 *   - @ref physical_object may be decomposed and sync between mpi processes;
 *   - The element value of PhysicalObject may be accessed through a index of discrete grid point in the domain
 *
 *
 *  ## Member types
 *   Member type	 			| Semantics
 *   ---------------------------|--------------
 *   domain_type				| Domain
 *   iterator_type				| iterator of element value
 *   range_type					| range of element value
 *
 *
 *
 *  ## Member functions
 *
 *  ### Constructor
 *
 *   Pseudo-Signature 	 			| Semantics
 *   -------------------------------|--------------
 *   `PhysicalObject()`						| Default constructor
 *   `~PhysicalObject() `					| destructor.
 *   `PhysicalObject( const PhysicalObject& ) `	| copy constructor.
 *   `PhysicalObject( PhysicalObject && ) `			| move constructor.
 *   `PhysicalObject( Domain & D ) `			| Construct a PhysicalObject on domain \f$D\f$.
 *
 *  ### Swap
 *    `swap(PhysicalObject & )`					| swap
 *
 *  ###  Fuctions
 *   Pseudo-Signature 	 			| Semantics
 *   -------------------------------|--------------
 *   `bool is_valid() `  			| _true_ if PhysicalObject is valid for accessing
 *   `update()`					| allocate memory
 *   `dataset()`					| return the data set of PhysicalObject
 *   `clear()`						| set value to zero, allocate memory if empty() is _true_
 *   `T properties(std::string name)const` | get properties[name]
 *   `properties(std::string name,T const & v) ` | set properties[name]
 *   `std::ostream& print(std::ostream & os) const` | print description to `os`
 *
 *  ### Element access
 *   Pseudo-Signature 				| Semantics
 *   -------------------------------|--------------
 *   `value_type & at(index_type s)`   			| access element on the grid points _s_ with bounds checking
 *   `value_type & operator[](index_type s) `  | access element on the grid points _s_as
 *
 *  @}
 **/

/**
 * @ingroup physical_object
 */
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

	PhysicalObject(const PhysicalObject&); // copy constructor.

	virtual std::string get_type_as_string() const=0;

	virtual DataSet dataset() const =0; //!< return the data set of PhysicalObject

	virtual bool is_valid()
	{
		return is_valid_;
	}

	virtual void sync()
	{

	}
	virtual void load()
	{
	}
	virtual bool update()
	{
		is_valid_ = true;
		return is_valid_;
	}

	virtual Properties const & properties(std::string const & name = "") const
	{
		return prop_[name];
	}

	virtual Properties & properties(std::string const & name = "")
	{
		return prop_[name];
	}

	virtual std::basic_ostream<char>& print(std::basic_ostream<char>& os) const
	{
		return prop_.print(os);

	}
private:
	Properties prop_;

	bool is_valid_ = false;

};
}  // namespace simpla

#endif /* CORE_PHYSICS_PHYSICAL_OBJECT_H_ */
