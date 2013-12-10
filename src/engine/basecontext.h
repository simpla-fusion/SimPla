/*
 * BaseContext.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef BASECONTEXT_H_
#define BASECONTEXT_H_

#include <cstddef>

namespace simpla
{
class LuaObject;

class BaseContext
{
	size_t step_count_;
public:
	BaseContext() :
			step_count_(0)
	{
	}
	virtual ~BaseContext()
	{
	}

	inline size_t GetStepCount() const
	{
		return step_count_;
	}
	virtual void Deserialize(LuaObject const & cfg)
	{
	}
	virtual void Serialize(LuaObject * cfg) const
	{
	}
	virtual void OneStep()
	{
		++step_count_;
	}
	virtual void DumpData()
	{
	}
};
}  // namespace simpla
#endif /* BASECONTEXT_H_ */
