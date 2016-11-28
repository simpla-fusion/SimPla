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

class Model :
        public concept::Printable,
        public geometry::GeoObject,
        public concept::Configurable
{
public:
    Model() {}

    virtual ~Model() {}

    virtual bool is_a(std::type_info const &info) const { return typeid(Model) == info; }

    virtual std::type_index typeindex() const { return std::type_index(typeid(Model)); }

    virtual std::string get_class_name() const { return "Model"; }

    virtual std::ostream &print(std::ostream &os, int indent) const { return os; }

    virtual void load(std::string const &);

    virtual void save(std::string const &);

    virtual void update();

    virtual void initialize(Real data_time = 0);

    void set_chart(std::shared_ptr<Chart> const &c) { m_chart_ = c; };

    std::shared_ptr<Chart> const &get_mesh() const { return m_chart_; };
private:
    std::shared_ptr<Chart> m_chart_;

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
