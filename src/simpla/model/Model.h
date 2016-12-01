//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/manifold/Chart.h>
#include <simpla/manifold/Bundle.h>
#include <simpla/geometry/GeoObject.h>

namespace simpla { namespace model
{
using namespace mesh;

class GeoObject;

class Model :
        public Object,
        public concept::Printable,
        public concept::Configurable,
        public concept::Deployable
{
public:

    enum MODEL_TAG { VACUUME = 1, PLASMA = 1 << 1, CUSTOM = 1 << 20 };

    SP_OBJECT_HEAD(Model, Object)

    Model(std::shared_ptr<Chart> const &c = nullptr);

    virtual ~Model();

    virtual void add_object(std::string const &name, std::shared_ptr<geometry::GeoObject> const &);

    virtual void remove_object(std::string const &key);

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual void load(std::string const &);

    virtual void save(std::string const &);

    virtual void deploy();

    virtual void preprocess();

    virtual void initialize(Real data_time = 0);

    virtual void next_time_step(Real data_time, Real dt);

    virtual void finalize(Real data_time = 0);

    virtual void postprocess();

    virtual mesh::EntityIdRange const &
    select(MeshEntityType iform, int tag);

    virtual mesh::EntityIdRange const &
    select(MeshEntityType iform, std::string const &tag);

    virtual mesh::EntityIdRange const &
    interface(MeshEntityType iform, const std::string &tag_in, const std::string &tag_out = "VACUUME");

    mesh::EntityIdRange const &interface(MeshEntityType iform, int tag_in, int tag_out);

    virtual mesh::EntityIdRange const &
    select(MeshEntityType iform, int tag) const
    {
        return m_range_cache_.at(iform).at(tag);
    }

    virtual mesh::EntityIdRange const &
    interface(MeshEntityType iform, int tag_in, int tag_out = VACUUME) const
    {
        return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
    }

    void set_chart(std::shared_ptr<Chart> const &c) { m_chart_ = c; };

    std::shared_ptr<Chart> const &get_mesh() const { return m_chart_; };
private:
    std::shared_ptr<Chart> m_chart_;

    Bundle<int, VERTEX, 9> m_tags_{m_chart_, "tags", "INPUT"};
    int m_g_obj_count_;
    std::map<std::string, int> m_g_name_map_;
    std::multimap<int, std::shared_ptr<geometry::GeoObject>> m_g_obj_;

    std::map<id_type, std::map<int, EntityIdRange>> m_range_cache_;
    std::map<id_type, std::map<int, std::map<int, EntityIdRange>>> m_interface_cache_;
//    virtual EntityIdRange
//    range(MeshEntityType const &iform, geometry::IntersectionStatus const &status, Chart const *c)
//    {
//        return EntityIdRange();
//    };
//
//    virtual void
//    get_value_fraction(MeshEntityType const &iform, Chart const *c, std::map<EntityIdRange, Real> *)
//    {
//        UNIMPLEMENTED;
//    }
//
//    virtual int const &tag(MeshEntityId const &s) const
//    {
//        return m_tags_.get(M::sw(M::minimal_vertex(s), s.w + M::node_id(s)));
//    }
//
//
//    virtual int const &fraction(MeshEntityId const &s) const
//    {
//        return m_tags_.get(M::sw(M::minimal_vertex(s), s.w + M::node_id(s)));
//    }
//
//
//    virtual int const &dual_fraction(MeshEntityId const &s) const
//    {
//        return m_tags_.get(M::sw(M::minimal_vertex(s), s.w + M::node_id(s)));
//    }
//
//protected:
//    virtual int &tag(MeshEntityId const &s)
//    {
//        return m_tags_.get(M::sw(M::minimal_vertex(s), s.w + M::node_id(s)));
//    }
//
//    virtual int &fraction(MeshEntityId const &s)
//    {
//        return m_tags_.get(M::sw(M::minimal_vertex(s), s.w + M::node_id(s)));
//    }
//
//    virtual int &dual_fraction(MeshEntityId const &s)
//    {
//        return m_tags_.get(M::sw(M::minimal_vertex(s), s.w + M::node_id(s)));
//    }





};
}}//namespace simpla{namespace model{

#endif //SIMPLA_MODEL_H
