/**
 * @file application.h
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_APPLICATION_H_
#define CORE_APPLICATION_APPLICATION_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "../gtl/design_pattern/SingletonHolder.h"
#include "../gtl/utilities/ConfigParser.h"

namespace simpla
{
/**
 *  @addtogroup task_flow Application
 *  @{
 *  @brief  framework of task_flow layer.
 */

struct SpApp
{
public:

    SpApp()
    {
    }

    virtual ~SpApp()
    {
    }

    virtual std::string const &description() = 0;

    virtual void body(ConfigParser &) = 0;
};

struct SpAppList : public std::map<std::string, std::shared_ptr<SpApp>>
{
};

template<typename T, typename ...Args>
std::string register_app(std::string const &name, Args &&...args)
{
    SingletonHolder<SpAppList>::instance()[name] = std::dynamic_pointer_cast<
            SpApp>(std::make_shared<T>(std::forward<Args>(args)...));

    return " ";
}

#define SP_APP(_app_name, _app_desc) \
struct _app_name:public SpApp  \
{ \
    static const std::string info; \
    typedef _app_name  this_type; \
    _app_name () {}\
    _app_name(this_type const &)=delete; \
    virtual ~_app_name  () {}\
    std::string const & description(){return info;} \
private:\
    void  body(ConfigParser & );\
};\
const std::string   _app_name::info =  register_app<_app_name>(( #_app_name))+_app_desc ; \
void _app_name::body(ConfigParser & options)

/** @} */

//#define USE_CASE(_use_case_name,_desc) SP_APP(_use_case_name,_desc)

}
// namespace simpla

#endif /* CORE_APPLICATION_APPLICATION_H_ */
