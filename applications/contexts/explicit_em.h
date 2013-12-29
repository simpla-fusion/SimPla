/*
 * \file explicit_em.h
 *
 *  Created on: 2013年12月12日
 *      Author: salmon
 */

#ifndef EXPLICIT_EM_H_
#define EXPLICIT_EM_H_

#include <memory>

namespace simpla
{
class BaseContext;
class LuaObject;

std::shared_ptr<BaseContext> CreateContextExplicitEM(LuaObject const &cfg);
}
// namespace simpla

#endif /* EXPLICIT_EM_H_ */
