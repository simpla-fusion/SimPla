//
// Created by salmon on 16-11-29.
//

#ifndef SIMPLA_DEPLOYABLE_H
#define SIMPLA_DEPLOYABLE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>

namespace simpla { namespace concept
{
struct LifeControllable
{

    LifeControllable() : m_is_deployed_(false), m_is_valid_(false) {}

    virtual ~LifeControllable() { destroy(); }

    virtual bool is_deployed() const { return m_is_deployed_; }

    virtual bool is_valid() const { return m_is_valid_; }

    /**
     * @name Life Cycle
     * @{
     */

    virtual void deploy()
    {
        if (is_deployed()) { RUNTIME_ERROR << "Repeat deploying!" << std::endl; }
        m_is_deployed_ = true;
    };


    virtual void pre_process() { m_is_valid_ = true; /*add sth here*/}

    virtual void post_process() {  /*add sth here*/ m_is_valid_ = false; }

    virtual void destroy() { m_is_deployed_ = false; };


    /** @}*/

private:
    bool m_is_deployed_ = false;
    bool m_is_valid_ = false;
};
}}//namespace simpla{namespace concept{
#endif //SIMPLA_DEPLOYABLE_H
