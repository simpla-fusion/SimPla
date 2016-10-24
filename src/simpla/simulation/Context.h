/**
 * @file context.h
 *
 * @date    2014-9-18  AM9:33:53
 * @author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <memory>
#include <list>
#include <map>

#include <simpla/SIMPLA_config.h>
#include <simpla/mesh/EntityRange.h>
#include <simpla/mesh/Patch.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/mesh/TransitionMap.h>
#include <simpla/mesh/DomainBase.h>
#include <simpla/toolbox/IOStream.h>


namespace simpla { namespace simulation
{

class ContextBase
{

public:

    ContextBase() {};

    virtual ~ContextBase() {};


    virtual void setup()=0;

    virtual void teardown()=0;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const =0;

    virtual toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const =0;

    virtual toolbox::IOStream &load(toolbox::IOStream &is)=0;

    virtual toolbox::IOStream &check_point(toolbox::IOStream &os) const =0;

    virtual std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb)=0;

    template<typename TProb, typename ...Args> std::shared_ptr<TProb>
    add_domain(Args &&...args)
    {
        auto res = std::make_shared<TProb>(std::forward<Args>(args)...);
        add_domain(res);
        return res;
    };

    virtual std::shared_ptr<mesh::DomainBase> get_domain(uuid id) const =0;

    template<typename TProb> std::shared_ptr<TProb>
    get_domain_as(uuid id) const
    {
        static_assert(!get_domain(id)->template is_a<TProb>(), "illegal type conversion!");
        assert(get_domain(id).get() != nullptr);
        return std::dynamic_pointer_cast<TProb>(get_domain(id));
    }

    virtual void sync(int level = 0, int flag = 0)=0;

    virtual void run(Real dt, int level = 0)=0;

    virtual Real time() const =0;

    virtual void time(Real t) =0;

    virtual void next_time_step(Real dt)=0;


};

class Context : public ContextBase
{
private:
    typedef Context this_type;
public:
    int m_refine_ratio = 2;

    Context();

    ~Context();


    void setup();

    void teardown();

    std::ostream &print(std::ostream &os, int indent = 1) const;

    toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const;

    toolbox::IOStream &load(toolbox::IOStream &is);

    toolbox::IOStream &check_point(toolbox::IOStream &os) const;

    std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb);

    std::shared_ptr<mesh::DomainBase> get_domain(uuid id) const;

    void sync(int level = 0, int flag = 0);

    void run(Real dt, int level = 0);

    Real time() const { return m_time_; }

    void time(Real t) { m_time_ = t; };

    void next_time_step(Real dt) { m_time_ += dt; };

private:
    Real m_time_;
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};


}}// namespace simpla{namespace simulation


#endif /* CORE_APPLICATION_CONTEXT_H_ */
