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

#include "../utilities/utilities.h"

#include "../gtl/design_pattern/singleton_holder.h"
#include "application.h"

namespace simpla
{
/** @ingroup application
 * @{
 *
 *  */
//
//class UseCase: public SpApp
//{
//	std::string case_info_;
//public:
//
//	ConfigParser options;
//
//	UseCase()
//	{
//	}
//	virtual ~UseCase()
//	{
//	}
//
//	virtual void init(int argc, char ** argv)
//	{
//		options.init(argc, argv);
//	}
//
//	virtual void body() =0;
//};
//
//struct UseCaseList
//{
//	std::map<std::string, std::shared_ptr<UseCase>> list_;
//
//	std::string add(std::string const & name,
//			std::shared_ptr<UseCase> const & p);
//
//	template<typename T>
//	std::string add(std::string const & name, std::shared_ptr<T> const & p)
//	{
//		list_[name] = std::dynamic_pointer_cast<UseCase>(p);
//		return "UseCase" + value_to_string(list_.size()) + "_" + name;
//	}
//	template<typename T>
//	std::string add(std::string const & name)
//	{
//		return add(name, std::make_shared<T>());
//	}
//
//	std::ostream & print(std::ostream & os);
//
//	void run(int argc, char ** argv);
//
//}
//;
//template<typename T>
//std::string RegisterUseCase(std::string const &name)
//{
//	return SingletonHolder<UseCaseList>::instance().template add < T > (name);
//}
//inline void RunAllUseCase(int argc, char ** argv)
//{
//	SingletonHolder<UseCaseList>::instance().run(argc, argv);
//}
#define USE_CASE(_use_case_name) SP_APP(_use_case_name)
#define SP_MAIN(_use_case_name) SP_APP(_use_case_name)

/** @} */

}  // namespace simpla

#endif /* CORE_APPLICATION_USE_CASE_H_ */
