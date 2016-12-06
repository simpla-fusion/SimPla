//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/toolbox/design_pattern/Observer.h>
#include <simpla/mesh/MeshBlock.h>
#include <simpla/mesh/Attribute.h>

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
struct Chart : public concept::Printable, public concept::Deployable
{
    Chart();

    virtual ~Chart();

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual std::type_index typeindex() const { return std::type_index(typeid(Chart)); }

    virtual std::string get_class_name() const { return "CoordinateFrame"; }

    virtual bool is_a(std::type_info const &info) const;

    virtual void pre_process();

    virtual void post_process();

    virtual void initialize(Real data_time = 0);

    virtual void finalize(Real data_time = 0);

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
    U const *mesh_as() const
    {
        ASSERT(this->is_a(typeid(U)));
        return static_cast<U const *>(this);
    }

    /**
     * @param attributes
     */
    AttributeView *connect(AttributeView *attr);

    void disconnect(AttributeView *attr);

    std::set<AttributeView *> &attributes();

    std::set<AttributeView *> const &attributes() const;

    virtual point_type point(index_type i, index_type j, index_type k) const { return m_mesh_block_->point(i, j, k); };

protected:
    std::set<AttributeView *> m_attr_views_;

    std::shared_ptr<MeshBlock> m_mesh_block_;

};


}}//namespace simpla { namespace mesh

#endif //SIMPLA_GEOMETRY_H
