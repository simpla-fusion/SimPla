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

namespace simpla
{
class UseCase
{
	int argc_ = 0;
	char ** argv_ = nullptr;
	LuaObject dict_;

	std::string case_info_;

public:
	UseCase()
	{
	}
	virtual ~UseCase()
	{
	}

	void parse_cmd_line(
			std::function<int(std::string const &, std::string const &)> const & fun);

	std::tuple<bool, std::string> cmdline_option(
			std::string const & name) const;

	template<typename T>
	std::tuple<bool, T> option(std::string const & name) const
	{
		bool is_found = false;
		std::string value_str;

		std::tie(is_found, value_str) = cmdline_option(name);

		if (is_found)
		{
			return std::make_tuple(true, ToValue<T>(value_str));
		}
		else if (dict_[name])
		{
			return std::make_tuple(true, dict_[name].template as<T>());
		}

		return std::make_tuple(false, T());
	}

	template<typename T>
	T option(std::string const & name, T const & default_value) const
	{
		bool is_found = false;
		T res;

		std::tie(is_found, res) = option<T>(name);

		if (is_found)
		{
			return res;
		}
		else
		{
			return default_value;
		}

	}

	void run(int argc, char ** argv);

	virtual void case_body()=0;
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

#define SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name) _parent_class_name##_##_use_case_name

#define SP_USE_CASE(_use_case_name,_parent_class_name,_parent_class) \
class SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name):public _parent_class  \
{ \
public:\
	typedef SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name)  this_type; \
	SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name) () {}\
	virtual ~SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name) () {}\
	SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name) (this_type const &)=delete; \
	static const std::string case_info; \
private:\
  virtual void case_body();\
};\
  const std::string \
SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name)::case_info = \
	use_case_register<SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name)>(( #_use_case_name)) ; \
void SP_USE_CASE_CLASS_NAME_(_use_case_name,_parent_class_name)::case_body()

}  // namespace simpla

#endif /* CORE_APPLICATION_USE_CASE_H_ */
