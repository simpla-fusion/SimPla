//
// Created by salmon on 17-10-16.
//

#ifndef SIMPLA_EBDOMAIN_H
#define SIMPLA_EBDOMAIN_H

#include "Domain.h"
namespace simpla {
namespace engine {

struct EBDomainBase {};

template <typename TChart, template <typename> class... Policies>
struct EBDomain : public Domain<TChart, Policies...>, public EBDomainBase {
    SP_SERIALIZABLE_HEAD(Domain, EBDomain)

   public:
    EBDomain();
    ~EBDomain() override;
    bool CheckBlockCrossBoundary() const override;

    template <template <typename> class U>
    std::shared_ptr<U<this_type>> AddEmbeddedDomain(std::string const &k,
                                                    std::shared_ptr<geometry::GeoObject> const &g) {
        auto res = U<this_type>::New();
        res->SetBoundary(g);
        //        DomainBase::AddEmbeddedDomain(k, res);
        return res;
    };
    engine::AttributeT<Real, NODE> m_vertex_tag_{this, "Name"_ = "vertex_tag"};
    //    Field<host_type, Real, EDGE> m_edge_tag_{m_domain_, "name"_ = "edge_tag"};
    //    Field<host_type, Real, NODE, 3> m_edge_tag_d_{m_domain_, "name"_ = "edge_tag_d"};
    //    Field<host_type, Real, FACE> m_face_tag_{m_domain_, "name"_ = "face_tag"};
    engine::AttributeT<Real, CELL> m_volume_tag_{this, "Name"_ = "volume_tag"};

    engine::AttributeT<unsigned int, NODE> m_node_tag_{this, "Name"_ = "node_tag"};
    engine::AttributeT<Real, EDGE> m_edge_frac_{this, "Name"_ = "edge_frac"};
    engine::AttributeT<Real, FACE> m_face_frac_{this, "Name"_ = "face_frac"};
    engine::AttributeT<Real, CELL> m_cell_frac_{this, "Name"_ = "cell_frac"};
};

template <typename TChart, template <typename> class... Policies>
EBDomain<TChart, Policies...>::EBDomain() {}

template <typename TChart, template <typename> class... Policies>
EBDomain<TChart, Policies...>::~EBDomain() {}

}  // namespace engine
}  // namespace simpla
#endif  // SIMPLA_EBDOMAIN_H
