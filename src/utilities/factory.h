/*
 * factory.h
 *
 *  Created on: 2014-6-13
 *      Author: salmon
 */

#ifndef FACTORY_H_
#define FACTORY_H_

#include <map>
#include "utilities.h"
#include "log.h"

namespace simpla
{

//! \ingroup Utilities

/**
 *  \cite  Modern C++ Design, Andrei Alexandrescu , Addison Wesley 2001  Charpt 8
 */
template<typename TId, typename TProduct, typename ...Args>
struct Factory
{
public:
	typedef TId identifier_type;

	typedef typename std::conditional<std::is_fundamental<TProduct>::value, TProduct, std::shared_ptr<TProduct>>::type product_type;

	typedef std::function<product_type(Args ...)> create_fun_callback;

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

	int Register(identifier_type const & id, create_fun_callback const &fun)
	{
		return callbacks_.insert(typename CallbackMap::value_type(id, fun)).second ? 1 : 0;
	}
	int Unregister(identifier_type const & id)
	{
		return callbacks_.erase(id);
	}
private:
	CallbackMap callbacks_;

}
;
}
// namespace simpla

#endif /* FACTORY_H_ */
