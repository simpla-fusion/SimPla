//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <memory>

#include <simpla/manifold/Atlas.h>
#include <simpla/data/DataBase.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Configurable.h>

namespace simpla { namespace simulation
{
class TimeIntegrator :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable
{
public:
    TimeIntegrator(std::string const &s_name = "") : Object(), m_name_(s_name) {}

    virtual ~TimeIntegrator() {}

    virtual void update() {};

    virtual void tear_down() {};

    virtual void register_worker(std::shared_ptr<mesh::Worker> const &w) { m_worker_ = w; }

    virtual std::shared_ptr<mesh::Worker> &worker() { return m_worker_; }

    virtual std::string name() const { return m_name_; };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; }

    virtual void load(data::DataBase const &) { UNIMPLEMENTED; };

    virtual void save(data::DataBase *) const { UNIMPLEMENTED; };

    virtual void update_level(int l0, int l1) { UNIMPLEMENTED; };

    virtual void advance(Real dt, int level = 0) { UNIMPLEMENTED; };

    virtual size_type next_step(Real dt) { UNIMPLEMENTED; };

    virtual void check_point() {};

    virtual size_type step() const { return 0; };

    virtual bool remaining_steps() const { return 0; };

    virtual Real time_now() const { return 0.0; }


private:
    std::string m_name_;
    std::shared_ptr<mesh::Worker> m_worker_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_TIMEINTEGRATOR_H
