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
    TimeIntegrator(std::string const &s_name);

    virtual ~TimeIntegrator();

    virtual void update_level(int l0, int l1);

    virtual void coarsen_level(int l);

    virtual void advance(Real dt, int level = 0);

    virtual std::string name() const;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void load(data::DataBase const &);

    virtual void save(data::DataBase *) const;

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;


};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_TIMEINTEGRATOR_H
