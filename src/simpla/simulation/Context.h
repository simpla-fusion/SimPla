/**
 * @file context.h
 *
 * @date    2014-9-18  AM9:33:53
 * @author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <memory>
#include <list>
#include <map>

#include <simpla/SIMPLA_config.h>
#include <simpla/mesh/EntityIdRange.h>
#include <simpla/mesh/Mesh.h>
#include <simpla/mesh/Patch.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/mesh/TransitionMap.h>
#include <simpla/mesh/DomainBase.h>
#include <simpla/toolbox/IOStream.h>
#include <simpla/toolbox/DataBase.h>


namespace simpla { namespace simulation
{
class WorkerBase;

/**
 *  life cycle of a simpla::Context
 *
 *
 *
 * constructure->initialize -> load -> +-[ add_*    ]  -> deploy-> +- [ next_step   ] -> save -> [ teardown ] -> destructure
 *                                     |-[ register*]              |- [ check_point ]
 *                                                                 |- [ print       ]
 *
 *
 *
 *
 *
 *
 */
class ContextBase : public toolbox::Object
{

public:

    ContextBase(std::string const &name_str = "") : toolbox::Object(name_str) {};

    virtual ~ContextBase() {};

    virtual void initialize(int argc = 0, char **argv = nullptr)=0;

    virtual void load(const std::shared_ptr<toolbox::DataBase> &) =0;

    virtual void save(toolbox::DataBase *) =0;

    virtual void deploy()=0;

    virtual void teardown()=0;

    virtual void registerAttribute(std::string const &, std::shared_ptr<mesh::AttributeBase> &, int flag = 0) =0;

    virtual bool is_valid() const { return true; };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

    virtual toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    virtual size_type step() const =0;

    virtual Real time() const =0;

    virtual void next_time_step(Real dt)=0;

    virtual void registerWorker(std::shared_ptr<WorkerBase>)=0;

    virtual void registerAttribute(std::shared_ptr<mesh::AttributeBase>)=0;

    template<typename TWorker> void registerWorker(std::shared_ptr<TWorker>);


};

class Context : public ContextBase
{
private:
    typedef Context this_type;
public:

    Context();

    ~Context();

    void setup();

    void teardown();

    std::ostream &print(std::ostream &os, int indent = 1) const;

    toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const;

    toolbox::IOStream &load(toolbox::IOStream &is);

    toolbox::IOStream &check_point(toolbox::IOStream &os) const;

//    std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb);
//
//    std::shared_ptr<mesh::DomainBase> get_domain(mesh_id_type id) const;

    void sync(int level = 0, int flag = 0);

    void run(Real dt, int level = 0);

    Real time() const { return m_time_; }

    void time(Real t) { m_time_ = t; };

    void next_time_step(Real dt) { m_time_ += dt; };

private:
    Real m_time_;
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};


}}// namespace simpla{namespace simulation


#endif /* CORE_APPLICATION_CONTEXT_H_ */
