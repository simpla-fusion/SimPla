//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <memory>

#include <simpla/mesh/Atlas.h>
#include <simpla/data/DataTable.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Configurable.h>
#include <simpla/concept/LifeControllable.h>

namespace simpla { namespace simulation
{
class TimeIntegrator :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable,
        public concept::LifeControllable
{
public:
    TimeIntegrator(std::shared_ptr<mesh::Worker> const &w = nullptr) : Object(), m_worker_(w) {}

    virtual ~TimeIntegrator() {}

    virtual std::shared_ptr<mesh::Worker> &worker() { return m_worker_; }

    virtual std::shared_ptr<mesh::Worker> const &worker() const { return m_worker_; }

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const { return os; }

    virtual void Load(data::DataTable const &) { UNIMPLEMENTED; };

    virtual void Save(data::DataTable *) const { UNIMPLEMENTED; };

    virtual void update_level(int l0, int l1) { UNIMPLEMENTED; };

    virtual void advance(Real dt, int level = 0) { UNIMPLEMENTED; };

    virtual size_type next_step(Real dt) { UNIMPLEMENTED; return 0;};

    virtual void check_point() {};

    virtual size_type step() const { return 0; };

    virtual bool remaining_steps() const { return 0; };

    virtual Real time_now() const { return 0.0; }


private:
    std::shared_ptr<mesh::Worker> m_worker_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_TIMEINTEGRATOR_H
