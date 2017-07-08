//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_STRUCTUREDMESH_H
#define SIMPLA_STRUCTUREDMESH_H

#include <simpla/algebra/Array.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/geometry/Chart.h>

#include "Mesh.h"
namespace simpla {
namespace mesh {

/**
 *  Structured Mesh
 *  - index space and local coordinates have same origin coordinates
 *
 */
class StructuredMesh : public MeshBase {
    SP_OBJECT_HEAD(StructuredMesh, MeshBase)
   public:
    static constexpr unsigned int NDIMS = 3;
    typedef Real scalar_type;
    typedef EntityId entity_id_type;
    typedef engine::Attribute attribute_type;

    template <typename V>
    using array_type = Array<V, ZSFC<NDIMS>>;

    template <typename V, int IFORM, int... DOF>
    using data_type = nTuple<array_type<V>, ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3), DOF...>;

    template <typename... Args>
    explicit StructuredMesh(Args &&... args) : MeshBase(std::forward<Args>(args)...){};

    ~StructuredMesh() override = default;

    SP_DEFAULT_CONSTRUCT(StructuredMesh);

    DECLARE_REGISTER_NAME(StructuredMesh);

    void DoUpdate() override;

    index_box_type GetIndexBox(int tag) const override;

    void AddGeometryObject(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix) override;

    point_type point(entity_id_type s) const;

    point_type local_coordinates(entity_id_type s, Real const *r = nullptr) const override;

    ZSFC<NDIMS> GetSpaceFillingCurve(int iform, int nsub = 0) const {
        return ZSFC<NDIMS>{GetIndexBox(EntityIdCoder::m_sub_index_to_id_[iform][nsub])};
    }

    EntityRange make_range(int IFORM) const {
        EntityRange res;
        int num_of_subs = ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
        for (int i = 0; i < num_of_subs; ++i) {
            int tag = EntityIdCoder::m_sub_index_to_id_[IFORM][i];
            auto id_box = GetIndexBox(tag);
            res.append(std::make_shared<ContinueRange<EntityId>>(id_box, tag));
        }

        return res;
    }

    void InitializeData(Real time_now) override { MeshBase::InitializeData(time_now); }
    void SetBoundaryCondition(Real time_now, Real time_dt) override {
        MeshBase::SetBoundaryCondition(time_now, time_dt);
    }

   protected:
    point_type m_dx_{1, 1, 1};
    point_type m_x0_{0, 0, 0};

   public:
    template <typename TL, typename TR>
    void DoFill(TL &lhs, TR const &rhs) const {
        CalculusPolicy<this_type>::Fill<simpla::traits::iform<TL>::value>(*this, lhs.Get(), rhs);
    }

    template <typename TL, typename... Args>
    decltype(auto) GetEntity(TL &lhs, Args &&... args) const {
        return CalculusPolicy<this_type>::GetEntity<simpla::traits::iform<TL>::value>(*this, lhs.Get(),
                                                                                      std::forward<Args>(args)...);
    }

    size_type GetNumberOfEntity(int IFORM = VERTEX) const {
        index_box_type m_index_box_ = GetBlock().GetIndexBox();
        return calculus::reduction<tags::multiplication>(std::get<1>(m_index_box_) - std::get<0>(m_index_box_)) *
               ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
    }
};
}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_STRUCTUREDMESH_H
