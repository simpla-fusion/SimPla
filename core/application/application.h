/*
 * application.h
 *
 *  Created on: 2014年11月21日
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_APPLICATION_H_
#define CORE_APPLICATION_APPLICATION_H_

namespace simpla
{
/**
 *  @addtogroup application Application
 *  @{
 *  @brief  framework of application layer.
 */

#define SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name) _parent_class_name##_##_use_case_name

#define SP_APP(_use_case_name,_parent_class_name,_parent_class) \
class SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name):public _parent_class  \
{ \
public:\
	typedef SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name)  this_type; \
	SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name) () {}\
	virtual ~SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name) () {}\
	SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name) (this_type const &)=delete; \
	static const std::string info; \
private:\
  virtual void  body();\
};\
  const std::string \
SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name)::info = \
	use_case_register<SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name)>(( #_use_case_name)) ; \
void SP_APP_CLASS_NAME_(_use_case_name,_parent_class_name)::body()

/** @} */

}  // namespace simpla

#endif /* CORE_APPLICATION_APPLICATION_H_ */
