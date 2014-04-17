/*
 * context.h
 *
 *  Created on: 2012-10-9
 *      Author: salmon
 */

#ifndef CONTEXT_H_
#define CONTEXT_H_

namespace simpla
{
class LuaObject;
struct Context
{
	std::function<void(LuaObject)> Load;

	std::function<void(std::string const &)> DumpData;

	std::function<std::ostream & (std::ostream &)> Save;

	std::function<void(double dt)> NextTimeStep;

};

inline std::ostream & operator<<(std::ostream & os, Context const &self)
{
	return self.Save(os);
}
}
// namespace simpla

#endif /* CONTEXT_H_ */
