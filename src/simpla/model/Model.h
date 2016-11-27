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

class Model : public concept::Printable
{
public:
    Model(ChartBase *c) : chart(c) {}

    virtual ~Model() {}

    virtual bool is_a(std::type_info const &info) const { return typeid(Model) == info; }

    virtual std::type_index typeindex() const { return std::type_index(typeid(Model)); }

    virtual std::string get_class_name() const { return "Model"; }

    virtual std::ostream &print(std::ostream &os, int indent) const { return os; }

    virtual void load(std::string const &);

    virtual void save(std::string const &);

    virtual void deploy();

    virtual void initialize(Real data_time = 0);


    enum { OUT = 0, IN = 1, CROSS_BORDER = 10 };

    typedef MeshEntityIdCoder M;


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

    ChartBase *chart;

private:
//    std::shared_ptr<geometry::GeoObject> m_geo_;
    Bundle<int, VERTEX, 9> m_tags_{chart, "tags", "INPUT"};
//    /**
//     *  0     : out
//     *  (0,1] : on the edge
//     *  1     : in
//     */
    Bundle<Real, VERTEX, 9> m_fraction_{chart, "volume_fraction", "INPUT"};
    Bundle<Real, VERTEX, 9> m_dual_fraction_{chart, "dual_volume_fraction", "INPUT"};


};
}}//namespace simpla{namespace model{

#endif //SIMPLA_MODEL_H
