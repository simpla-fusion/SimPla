/*
 * properties.h
 *
 *  Created on: 2012-3-6
 *      Author: salmon
 */

#ifndef PROPERTIES_H_
#define PROPERTIES_H_
#include "include/defs.h"
#include <string>
#include <map>
#include <boost/any.hpp>

class Properties: public std::map<std::string, boost::any>
{
public:
	typedef Properties ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

};

#endif /* PROPERTIES_H_ */
