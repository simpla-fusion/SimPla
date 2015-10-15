/**
 * @file sp_object.cpp
 *
 *  Created on: 2015-3-6
 *      Author: salmon
 */
#include "sp_object.h"


namespace simpla
{
//! Default constructor
SpObject::SpObject()
{

}

SpObject::SpObject(const SpObject &)
{
}

//! destroy.
SpObject::~SpObject()
{
}


std::ostream &SpObject::print(std::ostream &os) const
{
	return properties.print(os);
}
}  // namespace simpla
