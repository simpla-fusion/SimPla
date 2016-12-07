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

    /** @name excute cycle
     *  @{*/


    virtual void pre_process() { m_is_valid_ = true; /*add sth here*/}

    virtual void post_process() { /*add sth here*/ m_is_valid_ = false; }

    virtual void initialize(Real data_time = 0, Real dt = 0)
    {
        to_phase(0);
        pre_process();
    }

    virtual unsigned int
    to_phase(unsigned int phase_num)
    {
        m_current_phase_ = phase_num;
        return m_current_phase_;
    }

    virtual unsigned int
    next_phase(Real data_time = 0, Real dt = 0, unsigned int inc_phase = 1)
    {
        return to_phase(current_phase_num() + inc_phase);
    }

    virtual unsigned int current_phase_num() const { return m_current_phase_; }

    virtual unsigned int max_phase_num() const { return 1; };

    virtual void finalize(Real data_time = 0, Real dt = 0)
    {
        next_phase(data_time, dt, max_phase_num());
        post_process();
    }
    /** @}*/

private:
    bool m_is_deployed_ = false;
    bool m_is_valid_ = false;
    unsigned int m_current_phase_ = 0;
};
}}//namespace simpla{namespace concept{
#endif //SIMPLA_DEPLOYABLE_H
