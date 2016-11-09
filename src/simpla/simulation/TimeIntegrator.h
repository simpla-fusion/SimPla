//
// Created by salmon on 16-11-7.
//

#ifndef SIMPLA_TIMEINTEGRATOR_H
#define SIMPLA_TIMEINTEGRATOR_H

#include <simpla/SIMPLA_config.h>
#include <memory>

#include <simpla/mesh/Atlas.h>
#include <simpla/data/DataBase.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>

namespace simpla { namespace simulation
{
class TimeIntegrator :
        public Object,
        public concept::Printable,
        public concept::Serializable
{
public:
    TimeIntegrator(std::string const &s_name = "") : Object(s_name) {}

    virtual ~TimeIntegrator() {}

    virtual void deploy() {};

    virtual void tear_down() {};

    virtual void register_worker(std::shared_ptr<mesh::Worker> const &w) { m_worker_ = w; }

    virtual std::shared_ptr<mesh::Worker> &worker() { return m_worker_; }

    virtual std::string name() const { return Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; }

    virtual void load(data::DataBase const &) {};

    virtual void save(data::DataBase *) const {};

    virtual void next_time_step(Real dt) {};

    virtual void update_level(int l0, int l1) {};

    virtual void advance(Real dt, int level = 0) {};

    data::DataBase &config(std::string const &s = "") { return m_db_.get(s); }

    data::DataBase const &config(std::string const &s = "") const { return m_db_.at(s); }


    data::DataBase db;
//    data::DataBase &db(std::string const &s = "") { return m_db_.get(s); }
//
//    data::DataBase const &db(std::string const &s = "") const { return m_db_.at(s); }

private:
    std::shared_ptr<mesh::Worker> m_worker_;
    data::DataBase m_db_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_TIMEINTEGRATOR_H
