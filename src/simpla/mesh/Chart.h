//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/design_pattern/Observer.h>
#include <simpla/mesh/MeshBlock.h>

namespace simpla { namespace mesh
{
class Patch;

class DataBlock;

/**
 *  Define:
 *   A bundle is a triple $(E, p, B)$ where $E$, $B$ are sets and $p:E→B$ a map
 *   - $E$ is called the total space
 *   - $B$ is the base space of the bundle
 *   - $p$ is the projection
 *
 */
class Chart :
        public concept::Printable,
        public concept::LifeControllable
{
public:
SP_OBJECT_BASE(Chart);

    Chart();

    virtual ~Chart();

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual void accept(Patch *p);

    virtual void pre_process();

    virtual void post_process();

    virtual void initialize(Real data_time = 0, Real dt = 0);

    virtual void finalize(Real data_time = 0, Real dt = 0);

    virtual std::shared_ptr<MeshBlock> const &mesh_block() const { return m_mesh_block_; }


protected:
    std::shared_ptr<MeshBlock> m_mesh_block_;
};

template<typename ...> class ChartProxy;

template<typename U>
class ChartProxy<U> : public Chart, public U
{
    template<typename ...Args>
    explicit ChartProxy(Args &&...args):U(std::forward<Args>(args)...) {}

    ~ChartProxy() {}

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        U::print(os, indent);
        Chart::print(os, indent);
    }

    virtual void accept(Patch *p)
    {
        Chart::accept(p);
        U::accpt(p);
    };

    virtual void pre_process()
    {
        Chart::pre_process(p);
        U::pre_process(p);
    };

    virtual void post_process()
    {
        U::post_process(p);
        Chart::post_process(p);
    };

    virtual void initialize(Real data_time = 0, Real dt = 0)
    {
        Chart::initialize(data_time, dt);
        U::initialize(data_time, dt);
    }

    virtual void finalize(Real data_time = 0, Real dt = 0)
    {
        U::finalize(data_time, dt);
        Chart::finalize(data_time, dt);

    }
};

}}//namespace simpla { namespace mesh

#endif //SIMPLA_GEOMETRY_H
