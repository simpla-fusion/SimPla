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

#include "Mesh.h"


namespace simpla { namespace io { struct IOStream; }}

namespace simpla { namespace mesh
{
struct MeshAttributeBase;

class MeshBase;

class MeshWorker : public base::Object
{
public:

    SP_OBJECT_HEAD(MeshWorker, base::Object);

    MeshWorker() { }

    virtual  ~MeshWorker() { teardown(); }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual std::shared_ptr<MeshWorker> clone(MeshBase const &) const
    {
        UNIMPLEMENTED;
        return std::shared_ptr<MeshWorker>(nullptr);
    };

    virtual void update_ghost_from(MeshBase const &other) { };

    virtual bool same_as(MeshBase const &) const { return false; };

    virtual std::vector<box_type> refine_boxes() const { return std::vector<box_type>(); };

    virtual void refine(MeshBase const &other) { };

    virtual bool coarsen(MeshBase const &other) { return false; };

    virtual void setup() { };

    virtual void teardown() { };

    virtual io::IOStream &check_point(io::IOStream &os) const;

    virtual io::IOStream &save(io::IOStream &os) const;

    virtual io::IOStream &load(io::IOStream &is) const;


    virtual void next_step(Real dt)
    {
        m_time_ += dt;
        ++m_step_count_;
    }

    Real time() const { return m_time_; }

    size_t step_count() const { return m_step_count_; }

private:
    Real m_time_ = 0;
    size_t m_step_count_ = 0;
    std::map<std::string, std::shared_ptr<MeshAttributeBase> > m_attr_;
};
}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHWALKER_H
