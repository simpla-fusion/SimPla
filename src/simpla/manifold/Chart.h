//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/toolbox/design_pattern/Observer.h>
#include <simpla/mesh/MeshBlock.h>

namespace simpla { namespace mesh
{

/**
 *  Define:
 *   A bundle is a triple $(E, p, B)$ where $E$, $B$ are sets and $p:E→B$ a map
 *   - $E$ is called the total space
 *   - $B$ is the base space of the bundle
 *   - $p$ is the projection
 *
 */
struct Chart : public concept::Printable, public concept::LifeControllable
{
    Chart();

    virtual ~Chart();

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual std::type_index typeindex() const { return std::type_index(typeid(Chart)); }

    virtual std::string get_class_name() const { return "CoordinateFrame"; }

    virtual bool is_a(std::type_info const &info) const;

    virtual void pre_process();

    virtual void post_process();

    virtual void initialize(Real data_time = 0, Real dt = 0);

    virtual void finalize(Real data_time = 0, Real dt = 0);

    virtual void move_to(std::shared_ptr<MeshBlock> const &m);

    virtual std::shared_ptr<MeshBlock> const &mesh_block() const
    {
        ASSERT(m_mesh_block_ != nullptr);
        return m_mesh_block_;
    }

    /**
     * @return current MeshBlock
     */
    template<typename U>
    U const *as() const
    {
        ASSERT(this->is_a(typeid(U)));
        return static_cast<U const *>(this);
    }


    virtual point_type point(index_type i, index_type j, index_type k) const { return m_mesh_block_->point(i, j, k); };

protected:

    std::shared_ptr<MeshBlock> m_mesh_block_;

};


}}//namespace simpla { namespace mesh

#endif //SIMPLA_GEOMETRY_H
