//
// Created by salmon on 16-5-25.
//

#ifndef SIMPLA_METRICPOLICY_H
#define SIMPLA_METRICPOLICY_H
namespace simpla { namespace manifold { namespace policy
{

template<typename TMesh>
struct MetricPolicy
{

private:
    typedef TMesh mesh_type;

    typedef MetricPolicy<mesh_type> this_type;

    mesh_type const &m_mesh_;

public:
    typedef MetricPolicy<mesh_type> metric_policy;
    typename mesh::MeshEntityId id_type;

private:
    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(id_type s) const { return m_volume_[mesh_type::node_id(s)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_[mesh_type::node_id(s)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_[mesh_type::node_id(s)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_[mesh_type::node_id(s)]; }
};
}}//namespace simpla { namespace manifold { namespace policy

#endif //SIMPLA_METRICPOLICY_H
