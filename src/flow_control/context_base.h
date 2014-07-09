/**
 * context_base.h
 *
 * \date 2014年7月9日
 * \author salmon
 */

#ifndef CONTEXT_BASE_H_
#define CONTEXT_BASE_H_

#include <iostream>
#include <string>

namespace simpla
{

class ContextBase
{
public:
	ContextBase()
	{
	}

	virtual ~ContextBase()
	{
	}

	virtual std::string get_type_as_string() const =0;

	virtual std::string load(std::string const &) =0;

	virtual std::string save(std::string const &) const =0;

	virtual std::ostream & print(std::ostream &) const =0;

	virtual void next_timestep() =0;

	virtual bool pre_process() =0;

	virtual bool post_process() =0;

	virtual bool empty() const =0;

	virtual operator bool() const
	{
		return !empty();
	}
};

}  // namespace simpla

#endif /* CONTEXT_BASE_H_ */
