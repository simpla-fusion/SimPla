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


    /**
     * @name Life Cycle
     * @{
     */
    virtual bool is_deployed() const { return m_is_deployed_; }

    virtual void deploy()
    {
        if (is_deployed()) { RUNTIME_ERROR << "Repeat deploying!" << std::endl; }
        m_is_deployed_ = true;
    };

    virtual void destroy() { m_is_deployed_ = false; };

    /** @}*/

    virtual bool is_valid() const { return m_is_valid_; }

    virtual void sync(bool is_async = false) const {}

    /** @name execute cycle
     *  @{*/


    virtual void pre_process() { m_is_valid_ = true; /*add sth here*/}

    virtual void post_process() { next_phase();/*add sth here*/ m_is_valid_ = false; }

    virtual void phase(unsigned int num) { m_current_phase_ = num; }

    virtual unsigned int
    next_phase(unsigned int inc_phase = 0)
    {
        if (inc_phase == 0) { m_current_phase_ = max_phase_num(); } else { ++m_current_phase_; }
    }

    virtual unsigned int current_phase_num() const { return m_current_phase_; }

    virtual unsigned int max_phase_num() const { return 1; };


    /** @}*/

private:
    bool m_is_deployed_ = false;
    bool m_is_valid_ = false;
    unsigned int m_current_phase_ = 0;
};
}}//namespace simpla{namespace concept{
#endif //SIMPLA_DEPLOYABLE_H
