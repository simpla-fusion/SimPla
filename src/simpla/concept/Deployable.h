//
// Created by salmon on 16-11-29.
//

#ifndef SIMPLA_DEPLOYABLE_H
#define SIMPLA_DEPLOYABLE_H

#include <simpla/toolbox/Log.h>

namespace simpla { namespace concept
{
struct Deployable
{

    Deployable() {}

    virtual ~Deployable() { tear_down(); }

    bool is_deployed() const { return m_is_deployed_; }

    virtual void deploy()
    {
        if (is_deployed()) { RUNTIME_ERROR << "Repeat deploying!" << std::endl; }
        m_is_deployed_ = true;
    };

    virtual void tear_down() { m_is_deployed_ = false; };

private:
    bool m_is_deployed_ = false;

};
}}//namespace simpla{namespace concept{
#endif //SIMPLA_DEPLOYABLE_H
