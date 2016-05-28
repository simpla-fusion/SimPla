/** 
 * @file MeshWalker.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_MESHWALKER_H
#define SIMPLA_MESHWALKER_H

#include <memory>
#include "../gtl/primitives.h"
#include "Mesh.h"

namespace simpla { namespace mesh
{

class MeshBase;

class MeshWorker : public base::Object
{
public:

    SP_OBJECT_HEAD(MeshWorker, base::Object);

    MeshWorker() { }

    ~MeshWorker() { teardown(); }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual std::shared_ptr<MeshWorker> clone(MeshBase const &) const = 0;

    virtual void update_ghost_from(MeshBase const &const &other) = 0;

    virtual bool same_as(MeshBase const &) const = 0;

    virtual std::vector<box_type> refine_boxes() const = 0;

    virtual void refine(MeshBase const &const &other) = 0;

    virtual bool coarsen(MeshBase const &const &other) = 0;

    virtual void setup() { };

    virtual void teardown() { };

    virtual void check_point(io::IOStream &os) const { };

    virtual void save(io::IOStream &os) const { };

    virtual void load(io::IOStream &is) const { };

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
};
}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHWALKER_H
