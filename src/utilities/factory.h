/*
 * factory.h
 *
 *  Created on: 2014年6月13日
 *      Author: salmon
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <map>
#include "utilities.h"
#include "log.h"

namespace simpla
{
/**
 *  @ref Modern C++ Design, Andrei Alexandrescu , Addison Wesley 2001  Charpt 8
 */
template<typename TId, typename TProdcut, typename ...Args>
struct Factory
{
public:
	typedef TId identifier_type;
	typedef TProdcut product_type;
	typedef std::function<product_type(Args...)> create_fun_callback;

	typedef std::map<identifier_type, create_fun_callback> CallbackMap;

	Factory()
	{
	}
	;

	~Factory()
	{
	}

	product_type Create(identifier_type const &id, Args ... args) const
	{
		auto it = callbacks_.find(id);

		if (it == callbacks_.end())
		{
			RUNTIME_ERROR("Can not find id " + ToString(id));
		}
		return (it->second)(std::forward<Args>(args)...);
	}

	bool Register(identifier_type const & id, create_fun_callback const &fun)
	{
		return callbacks_.insert(typename CallbackMap::value_type(id, fun)).second;
	}
	bool Unregister(identifier_type const & id)
	{
		return callbacks_.erase(id) == 1;
	}
private:
	CallbackMap callbacks_;

}
;
}
// namespace simpla

#endif /* FACTORY_H_ */
