/**
 * \file context.h
 *
 * \date    2014年9月18日  上午9:33:53 
 * \author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <list>

#include "application.h"

namespace simpla
{
namespace _impl
{

class split;

}  // namespace _impl

class Context: public SpApp
{
public:

	Context(Context &, _impl::split)
	virtual ~Context();
	virtual void setup(int argc, char const** argv);
	virtual void teardown();
	virtual void task_schedule();

	struct Task
	{
		Task();
		virtual ~Task();
		virtual void body(Context & context);
	};

	void add_task(std::string const & name, std::shared_ptr<Task> const& p)
	{
		task_graph_.emplace_back(name, p);
	}
	void body()
	{
		for (auto & item : task_graph_)
		{
			item.second->body(*this);
		}
	}
private:
	std::list<std::pair<std::string, std::shared_ptr<Task>>>task_graph_;
};

template<typename TContext>
std::string add_task(std::string const & ctx_name,
		std::string const & task_name,
		std::shared_ptr<Context::Task> const & task)
{
	auto & app_list = SingletonHolder<SpAppList>::instance();
	if (app_list.find(ctx_name) == app_list.end())
	{
		register_app<TContext>(ctx_name);
	}
	std::dynamic_pointer_cast<TContext>(app_list[ctx_name]).add_task(task_name,
			task);

	return ctx_name + "." + task_name;
}
#define REGISTER_CONTEXT(_name,_type)  \
const static std::string  _name##_type_str = register_app<_type>(( #_name)) ;

#define ADD_TASK(_ctx_type,_ctx_name,_task_name)     namespace _impl{                  \
class _ctx_name##_task_name: public Context::Task                                      \
{                                                                                      \
	static const std::string info;                                                     \
public:                                                                                \
	typedef _ctx_name##_task_name this_type;                                           \
                                                                                       \
	void body(Context & context);                                                      \
	{                                                                                  \
		body2( dynamic_cast<_ctx_type&>(context));                                     \
	}                                                                                  \
	void body2(_ctx_type & context);                                                   \
};                                                                                     \
const std::string _ctx_name##_task_name::info =                                        \
add_task<_ctx_type>((#_ctx_name, #_task_name,std::make_shared<_ctx_name##_task_name>())) ;   \
} \
void _impl::_ctx_name##_task_name::body2(_ctx_type & context)

}
// namespace simpla

#endif /* CORE_APPLICATION_CONTEXT_H_ */
