//
// Created by salmon on 17-7-8.
//

#ifndef SIMPLA_DUMMYMESH_H
#define SIMPLA_DUMMYMESH_H
namespace simpla {

template <typename THost>
struct DummyDomain : public MeshBase {
   public:
    typedef DummyDomain this_type;

    point_type m_dx_{1, 1, 1};
    point_type m_x0_{0, 0, 0};
    static constexpr unsigned int NDIMS = 3;

    typedef EntityId entity_id_type;
    typedef DummyAttribute<DummyDomain> attribute_type;
    template <typename V, int IFORM, int... DOF>
    using data_type = nTuple<Array<V, ZSFC<NDIMS>>, (IFORM == NODE || IFORM == CELL) ? 1 : 3, DOF...>;
    index_box_type m_index_box_;

    typedef DummyDomain mesh_type;
    explicit DummyDomain(index_box_type const &i_box) : m_index_box_(i_box) {}
    ~DummyDomain() = default;

    ZSFC<3> GetSpaceFillingCurve(int IFORM, int sub = 0) const { return ZSFC<3>(m_index_box_); }

    template <typename TFun>
    void Foreach(TFun const &fun, size_type iform = NODE, size_type dof = 1) const {}
    Real volume(EntityId s) const { return 1.0; }
    Real dual_volume(EntityId s) const { return 1.0; }
    Real inv_volume(EntityId s) const { return 1.0; }
    Real inv_dual_volume(EntityId s) const { return 1.0; }

    point_type point(EntityId s) const {
        return point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)};
    }
    point_type point(EntityId s, point_type const &pr) const {
        return point_type{static_cast<Real>(s.x) + pr[0], static_cast<Real>(s.y) + pr[1],
                          static_cast<Real>(s.z) + pr[2]};
    };

    point_type map(point_type const &x) const { return x; }
    point_type inv_map(point_type const &x) const { return x; }
    void SetOrigin(point_type x) { m_x0_ = x; }
    void SetDx(point_type dx) { m_dx_ = dx; }
    point_type const &GetOrigin() { return m_x0_; }
    point_type const &GetDx() { return m_dx_; };

    this_type *GetMesh() { return this; };
    this_type const *GetMesh() const { return this; };

    virtual index_box_type IndexBox(int tag = NODE) const { return m_index_box_; }
    virtual point_type local_coordinates(EntityId s, Real const *r) const { return point_type{0, 0, 0}; };

    template <typename TL, typename TR>
    void Fill(TL &lhs, TR const &rhs) const {
        CalculusPolicy<this_type>::Fill<traits::iform<TL>::value>(*this, lhs.get(), rhs);
    }

    template <typename TL, typename... Args>
    decltype(auto) GetEntity(TL &lhs, Args &&... args) const {
        return CalculusPolicy<this_type>::GetEntity<traits::iform<TL>::value>(*this, lhs.get(),
                                                                              std::forward<Args>(args)...);
    }

    size_type GetNumberOfEntity(int IFORM = NODE) const {
        return calculus::reduction<tags::multiplication>(std::get<1>(m_index_box_) - std::get<0>(m_index_box_)) *
               ((IFORM == NODE || IFORM == CELL) ? 1 : 3);
    }
    bool empty() const { return false; }
};
}
#endif  // SIMPLA_DUMMYMESH_H