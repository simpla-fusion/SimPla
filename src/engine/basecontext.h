/*
 * BaseContext.h
 *
 *  Created on: 2012-10-26
 *      Author: salmon
 */

#ifndef BASECONTEXT_H_
#define BASECONTEXT_H_

#include <cstddef>
#include <iostream>
#include <string>

namespace simpla
{
class LuaObject;

class BaseContext
{
	size_t step_count_;
public:

	friend std::ostream & operator<<(std::ostream & os,
			BaseContext const &self);

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
	virtual void DumpData() const
	{
	}
	virtual std::ostream & Serialize(std::ostream & os) const
	{
		return os;
	}
	virtual void NextTimeStep()
	{
		++step_count_;
	}
};

std::ostream & operator<<(std::ostream & os, BaseContext const &self)
{

	return self.Serialize(os);
}

}  // namespace simpla
#endif /* BASECONTEXT_H_ */
