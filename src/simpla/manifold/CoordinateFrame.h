//
// Created by salmon on 16-11-24.
//

#ifndef SIMPLA_FRAMEBUNDLE_H
#define SIMPLA_FRAMEBUNDLE_H


namespace simpla { namespace mesh
{
struct ChartBase;

struct CoordinateFrame
{
    CoordinateFrame(ChartBase *c) : chart(c) {}

    virtual ~CoordinateFrame() {}

    virtual bool is_a(std::type_info const &info) const { return info == typeid(CoordinateFrame); }

    virtual void move_to(std::shared_ptr<MeshBlock> const &m) { m_mesh_block_ = m; };

    virtual MeshBlock const *mesh_block() const { return m_mesh_block_.get(); }

    virtual void deploy() {};

    virtual void initialize() {};

    virtual point_type point(MeshEntityId s) const =0;

    virtual point_type point(MeshEntityId s, point_type const &r) const =0;

    virtual Real volume(MeshEntityId s) const =0;

    virtual Real dual_volume(MeshEntityId s) const =0;

    virtual Real inv_volume(MeshEntityId s) const =0;

    virtual Real inv_dual_volume(MeshEntityId s) const =0;

    virtual id_type id() const { return m_mesh_block_->id(); }

    virtual bool is_inside(point_type const &p) const { return m_mesh_block_->is_inside(p); }

    virtual bool is_inside(index_tuple const &p) const { return m_mesh_block_->is_inside(p); }

    ChartBase *chart;

    std::shared_ptr<MeshBlock> m_mesh_block_;
};
}} //namespace simpla {

#endif //SIMPLA_FRAMEBUNDLE_H
