/*
 * application.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_APPLICATION_H_
#define CORE_APPLICATION_APPLICATION_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "../design_pattern/singleton_holder.h"
#include "../utilities/config_parser.h"

namespace simpla
{
/**
 *  @addtogroup application Application
 *  @{
 *  @brief  framework of application layer.
 */

struct SpApp
{
	std::string case_info_;
public:

	ConfigParser options;

	SpApp()
	{
	}
	virtual ~SpApp()
	{
	}

	virtual void setup(int argc, char ** argv)
	{
		options.init(argc, argv);
	}

	virtual void body() =0;
};
struct SpAppList: public std::map<std::string, std::shared_ptr<SpApp>>
{

	std::string add(std::string const & name, std::shared_ptr<SpApp> const & p);

	std::ostream & print(std::ostream & os);

	void run(int argc, char ** argv);
}
;

template<typename T, typename ...Args>
std::string register_app(std::string const & name, Args && ...args)
{
	return SingletonHolder<SpAppList>::instance().add(name,
			std::dynamic_pointer_cast<SpApp>(
					std::make_shared<T>(std::forward<Args>(args)...)));
}

inline void run_all_apps(int argc, char ** argv)
{
	SingletonHolder<SpAppList>::instance().run(argc, argv);
}

#define SP_APP(_app_name) \
class _app_name:public SpApp  \
{ \
	static const std::string info; \
public:\
	typedef _app_name  this_type; \
	_app_name () {}\
	_app_name(this_type const &)=delete; \
	virtual ~_app_name  () {}\
private:\
  virtual void  body();\
};\
const std::string   _app_name::info =  register_app<_app_name>(( #_app_name)) ; \
void _app_name::body()

/** @} */

}
// namespace simpla

#endif /* CORE_APPLICATION_APPLICATION_H_ */
