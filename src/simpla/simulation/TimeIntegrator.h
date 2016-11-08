//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <memory>

#include <simpla/mesh/Atlas.h>

namespace simpla { namespace simulation
{
class TimeIntegrator :
        public toolbox::Object,
        public toolbox::Printable,
        public toolbox::Serializable
{
public:
    TimeIntegrator(std::string const &s_name) : toolbox::Object(s_name) {}

    virtual ~TimeIntegrator() {}

    virtual void registerWorker(std::shared_ptr<mesh::Worker> const &w) { m_worker_ = w; }

    virtual std::shared_ptr<mesh::Worker> &worker() { return m_worker_; }

    virtual std::string name() const { return toolbox::Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; }

    virtual void load(data::DataBase const &) {};

    virtual void save(data::DataBase *) const {};

    virtual void update_level(int l0, int l1) {};

    virtual void advance(Real dt, int level = 0) {};

private:
    std::shared_ptr<mesh::Worker> m_worker_;
};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_TIMEINTEGRATOR_H
