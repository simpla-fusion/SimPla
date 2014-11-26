/*
 * use_case.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_USE_CASE_H_
#define CORE_APPLICATION_USE_CASE_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "../utilities/lua_state.h"
#include "../utilities/singleton_holder.h"
#include "../utilities/utilities.h"
#include "../utilities/optional.h"
#include "../utilities/config_parser.h"
#include "application.h"

namespace simpla
{
class UseCase
{
	std::string case_info_;
public:

	ConfigParser options;

	UseCase()
	{
	}
	virtual ~UseCase()
	{
	}

	virtual void init(int argc, char ** argv)
	{
		options.init(argc, argv);
	}

	virtual void body() =0;
};

struct UseCaseList
{
	std::map<std::string, std::shared_ptr<UseCase>> list_;

	std::string add(std::string const & name,
			std::shared_ptr<UseCase> const & p);

	template<typename T>
	std::string add(std::string const & name, std::shared_ptr<T> const & p)
	{
		list_[name] = std::dynamic_pointer_cast<UseCase>(p);
		return "UseCase" + ToString(list_.size()) + "_" + name;
	}
	template<typename T>
	std::string add(std::string const & name)
	{
		return add(name, std::make_shared<T>());
	}

	std::ostream & print(std::ostream & os);

	void run_all_case(int argc, char ** argv);

}
;

template<typename T>
std::string use_case_register(std::string const &name)
{
	return SingletonHolder<UseCaseList>::instance().template add<T>(name);
}
inline void RunAllUseCase(int argc, char ** argv)
{
	SingletonHolder<UseCaseList>::instance().run_all_case(argc, argv);
}

#define USE_CASE(_use_case_name) SP_APP(_use_case_name,usecase,UseCase)

}  // namespace simpla

#endif /* CORE_APPLICATION_USE_CASE_H_ */
