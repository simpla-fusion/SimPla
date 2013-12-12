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
	size_t step_count_;
	double sim_clock_;
public:

	std::string description;

	friend std::ostream & operator<<(std::ostream & os,
			BaseContext const &self);

	BaseContext() :
			step_count_(0), sim_clock_(0)
	{
	}
	virtual ~BaseContext()
	{
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
	virtual void NextTimeStep(double dt =
			std::numeric_limits<double>::quiet_NaN())
	{
		if (!std::isnan(dt))
		{
			sim_clock_ += dt;
		}

		++step_count_;
	}

	size_t GetStepCount() const
	{
		return step_count_;
	}

	double GetTime() const
	{
		return sim_clock_;
	}

	void SetTime(double clock)
	{
		sim_clock_ = clock;
	}

};

std::ostream & operator<<(std::ostream & os, BaseContext const &self)
{

	return self.Serialize(os);
}

}  // namespace simpla
#endif /* BASECONTEXT_H_ */
