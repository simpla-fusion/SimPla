/*
 * BaseContext.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef BASECONTEXT_H_
#define BASECONTEXT_H_

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

namespace simpla
{
class LuaObject;

class BaseContext
{

};

inline std::ostream & operator<<(std::ostream & os, BaseContext const &self)
{
	return self.Save(os);
}

}  // namespace simpla
#endif /* BASECONTEXT_H_ */
