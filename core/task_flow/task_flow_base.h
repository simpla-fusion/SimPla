/**
 * @file task_flow_base.h
 * @author salmon
 * @date 2015-11-21.
 */

#ifndef SIMPLA_TASK_FLOW_BASE_H
#define SIMPLA_TASK_FLOW_BASE_H
namespace simpla { namespace task_flow
{

class ContextBase;

struct TaskBase
{

    virtual void visit(ContextBase &ctx) { };

    virtual bool check_context_type(std::type_info const &) = 0;
};

class ContextBase
{
private:
    typedef ContextBase this_type;
public:

    ContextBase() { };

    virtual ~ContextBase() { };

    virtual void split(ContextBase &) { }

    virtual void setup(int argc, char **argv) { };

    virtual void tear_down() { };

    virtual void accept(TaskBase &v) { v.visit(*this); };

    virtual bool check_type(std::type_info const &) = 0;

//    virtual void accept(TaskBase &v) const { v.visit(*this); };
//    virtual void accept(TaskBase const &v) { v.visit(*this); };
//    virtual void accept(TaskBase const &v) const { v.visit(*this); };

};

}}//namespace simpla { namespace task_flow
#endif //SIMPLA_TASK_FLOW_BASE_H
