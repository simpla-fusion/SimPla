/** 
 * @file MeshWalker.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_MESHWALKER_H
#define SIMPLA_MESHWALKER_H

#include <memory>
#include "../base/Object.h"
#include "../gtl/primitives.h"
#include "../gtl/Log.h"

#include "../mesh/Mesh.h"


namespace simpla { namespace io { struct IOStream; }}

namespace simpla { namespace mesh
{
struct MeshAttributeBase;

class MeshBase;
}}


namespace simpla { namespace task_flow
{


class Worker : public base::Object
{
public:

    SP_OBJECT_HEAD(Worker, base::Object);

    Worker();

    virtual  ~Worker()noexcept;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual std::shared_ptr<Worker> clone(mesh::MeshBase const &) const
    {
        UNIMPLEMENTED;
        return std::shared_ptr<Worker>(nullptr);
    };

    virtual bool view(mesh::MeshBase const &other) { return false; };

    virtual void update_ghost_from(mesh::MeshBase const &other) { };

    virtual bool same_as(mesh::MeshBase const &) const { return false; };

    virtual std::vector<mesh::box_type> refine_boxes() const { return std::vector<mesh::box_type>(); };

    virtual void refine(mesh::MeshBase const &other) { };

    virtual bool coarsen(mesh::MeshBase const &other) { return false; };

    virtual void setup() { };

    virtual void teardown() { };

    virtual io::IOStream &check_point(io::IOStream &os) const;

    virtual io::IOStream &save(io::IOStream &os) const;

    virtual io::IOStream &load(io::IOStream &is) const;

    virtual void view(mesh::MeshBlockId const &) { }

    virtual void next_step(Real dt)
    {
        m_time_ += dt;
        ++m_step_count_;
    }

    template<typename T>
    T create(std::string const &s) const
    {
        return T();
    }

    Real time() const { return m_time_; }

    size_t step_count() const { return m_step_count_; }

private:
    Real m_time_ = 0;
    size_t m_step_count_ = 0;
    std::map<std::string, std::shared_ptr<mesh::MeshAttributeBase> > m_attr_;
};
}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHWALKER_H
