//
// Created by salmon on 16-11-24.
//

#ifndef SIMPLA_FRAMEBUNDLE_H
#define SIMPLA_FRAMEBUNDLE_H


namespace simpla { namespace mesh
{
struct ChartBase;

struct CoordinateFrame : public concept::Printable
{


    CoordinateFrame(ChartBase *c) : chart(c) {}

    virtual ~CoordinateFrame() {}

    virtual bool is_a(std::type_info const &info) const { return typeid(CoordinateFrame) == info; }

    virtual std::type_index typeindex() const { return std::type_index(typeid(CoordinateFrame)); }

    virtual std::string get_class_name() const { return "CoordinateFrame"; }


    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        os << "Type = \"" << get_class_name() << "\",";
        if (mesh_block() != nullptr)
        {
            os << std::endl;
            os << std::setw(indent + 1) << " " << " Block = {";
            mesh_block()->print(os, indent + 1);
            os << std::setw(indent + 1) << " " << "},";
        }
        return os;
    }

    virtual void move_to(std::shared_ptr<MeshBlock> const &m) { m_mesh_block_ = m; };

    virtual std::shared_ptr<MeshBlock> const &mesh_block() const { return m_mesh_block_; }

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
