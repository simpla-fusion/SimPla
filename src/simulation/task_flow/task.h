/**
 * @file  task.h
 *
 *  Created on: 2015-1-4
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_TASK_H_
#define CORE_APPLICATION_TASK_H_

#include <assert.h>
#include "../toolbox/type_traits.h"
#include "../toolbox/utilities/Log.h"
#include "task_flow_base.h"

namespace simpla { namespace task_flow
{


template<typename TContext>
class Task : public TaskBase
{
public:
    typedef TContext context_type;

    Task() { };

    virtual ~Task() { };

    SIMPLA_DISALLOW_COPY_AND_ASSIGN(Task)


    virtual bool check_context_type(std::type_info const &info)
    {
        return info == typeid(context_type);
    };

    virtual void visit(context_type &ctx) { UNIMPLEMENTED; }

    virtual void visit(ContextBase &ctx)
    {
        assert(ctx.check_type(typeid(context_type)));

        visit(dynamic_cast<context_type &>(ctx));
    }

};
namespace _impl
{

template<typename TContext>
struct TaskRegistry
{
    typedef TContext context_type;

    typedef Task<context_type> task_type;

    static std::map<std::string, std::shared_ptr<task_type>> m_registered_task_;

    static bool register_task(std::string const &k, std::shared_ptr<task_type> t)
    {
        bool success = false;

        std::tie(std::ignore, success) = m_registered_task_.insert(std::make_pair(k, t));

        if (!success) { VERBOSE << "Register task " << k << " fail!" << std::endl; }

        return success;
    }

    virtual std::shared_ptr<TaskBase> at(std::string const &k) const
    {
        return std::dynamic_pointer_cast<TaskBase>(m_registered_task_.at(k));
    };

};

template<typename TContext> std::map<std::string, std::shared_ptr<Task<TContext>>> TaskRegistry<TContext>::m_registered_task_;

}


#define SIMPLA_TASK_CLASS_NAME_(context_name, task_name)   context_name##_##task_name##_task

#define TASK(_CONTEXT_NAME_, _TASK_NAME_)                                                \
class SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_) : public Task<_CONTEXT_NAME_> \
{                                                                                        \
public:                                                                                  \
    SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_)() { }                           \
                                                                                         \
    virtual ~SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_)() { }                  \
                                                                                         \
    SIMPLA_DISALLOW_ASSIGN(SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_))         \
                                                                                         \
    virtual void visit(context_type &ctx);                                               \
                                                                                         \
    static bool is_registered(){return m_is_registered_;}                                \
                                                                                         \
    static bool add_to_registry()                                                        \
    {                                                                                    \
      return SingletonHolder<_impl::TaskRegistry<context_type>>::instance()              \
             .register_task( __STRING(_TASK_NAME_),                                      \
              std::dynamic_pointer_cast<task_base_type>(std::make_shared<this_type>())); \
    }                                                                                    \
private:                                                                                 \
                                                                                         \
    typedef Task<_CONTEXT_NAME_> task_base_type;                                         \
    typedef SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_) this_type;              \
    static bool m_is_registered_;                                                        \
};                                                                                       \
bool SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_,  _TASK_NAME_)::m_is_registered_ =           \
SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_)::add_to_registry();                 \
void SIMPLA_TASK_CLASS_NAME_(_CONTEXT_NAME_, _TASK_NAME_)::visit(_CONTEXT_NAME_ &ctx)


} // namespace task_flow
} // namespace simpla

#endif /* CORE_APPLICATION_TASK_H_ */
