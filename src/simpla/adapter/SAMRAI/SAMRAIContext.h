//
// Created by salmon on 16-10-27.
//

#ifndef SIMPLA_SAMRAICONTEXT_H
#define SIMPLA_SAMRAICONTEXT_H

#include <simpla/simulation/Context.h>
#include <simpla/simulation/Worker.h>

namespace simpla { namespace simulation
{

struct SAMRAIContext : public ContextBase
{
    SAMRAIContext();

    virtual ~SAMRAIContext();

    void setup(int argc, char *argv[]);

    void deploy();

    void teardown();

    void registerWorker(std::string const &name, std::shared_ptr<WorkerBase> const &);

    std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

    toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const { return os; };

    toolbox::IOStream &load(toolbox::IOStream &is) { return is; };

    toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb) {};

    std::shared_ptr<mesh::DomainBase> get_domain(uuid id) const {};

    size_type step() const;

    Real time() const;

    void next_time_step(Real dt);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}}//namespace simpla

#endif //SIMPLA_SAMRAICONTEXT_H
