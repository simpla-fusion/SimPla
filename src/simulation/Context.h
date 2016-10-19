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

#include "SIMPLA_config.h"

#include "mesh/EntityRange.h"
#include "mesh/Attribute.h"
#include "mesh/Atlas.h"
#include "mesh/TransitionMap.h"
#include "toolbox/IOStream.h"
#include "DomainBase.h"


namespace simpla { namespace simulation
{


class DomainBase;

class Context
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

    std::shared_ptr<DomainBase> add_domain(std::shared_ptr<DomainBase> pb);

    template<typename TProb, typename ...Args> std::shared_ptr<TProb>
    add_domain(Args &&...args)
    {
        auto res = std::make_shared<TProb>(std::forward<Args>(args)...);
        add_domain(res);
        return res;
    };

    std::shared_ptr<DomainBase> get_domain(uuid id) const;

    template<typename TProb> std::shared_ptr<TProb>
    get_domain_as(uuid id) const
    {
        static_assert(!get_domain(id)->is_a<TProb>(), "illegal type conversion!");
        assert(get_domain(id).get() != nullptr);
        return std::dynamic_pointer_cast<TProb>(get_domain(id));
    }

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
