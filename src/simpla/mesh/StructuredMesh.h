//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_STRUCTUREDMESH_H
#define SIMPLA_STRUCTUREDMESH_H

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Array.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/data/Data.h"
#include "simpla/engine/Attribute.h"
#include "simpla/geometry/Chart.h"
namespace simpla {
namespace mesh {

/**
 *  Structured Mesh
 *  - index space and local coordinates have same origin coordinates
 *
 */
class StructuredMesh {
   public:
    static constexpr unsigned int NDIMS = 3;
    typedef EntityId entity_id_type;

    StructuredMesh() = default;
    virtual ~StructuredMesh() = default;
    StructuredMesh(StructuredMesh const &) = delete;
    StructuredMesh(StructuredMesh &&) = delete;
    StructuredMesh &operator=(StructuredMesh const &) = delete;
    StructuredMesh &operator=(StructuredMesh &&) = delete;

    virtual std::shared_ptr<const geometry::Chart> GetChart() const = 0;
    virtual std::shared_ptr<const engine::MeshBlock> GetBlock() const = 0;
    virtual index_box_type IndexBox(int tag) const;

    ZSFC<NDIMS> GetSpaceFillingCurve(int iform, int nsub = 0) const {
        return ZSFC<NDIMS>{IndexBox(EntityIdCoder::m_sub_index_to_id_[iform][nsub])};
    }

    template <typename U>
    using array_type = Array<U, ZSFC<NDIMS>>;

    template <int IFORM, typename U, int... N>
    void initialize_data(nTuple<array_type<U>, N...> *d) const {
        traits::foreach (*d, [&](auto &a, int n0, auto &&... idx) {
            a.reset(IndexBox(EntityIdCoder::m_sub_index_to_id_[IFORM][n0]));
        });
    };

   public:
    size_type GetNumberOfEntity(int IFORM = NODE) const {
        index_box_type m_index_box_ = GetBlock()->IndexBox();
        return calculus::reduction<tags::multiplication>(std::get<1>(m_index_box_) - std::get<0>(m_index_box_)) *
               ((IFORM == NODE || IFORM == CELL) ? 1 : 3);
    }
};
}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_STRUCTUREDMESH_H
