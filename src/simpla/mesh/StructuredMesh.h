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

    point_type map(point_type const &p) const;
    point_type inv_map(point_type const &p) const;

    virtual const geometry::Chart &GetChart() const = 0;
    virtual const engine::MeshBlock &GetBlock() const = 0;

    point_type GetCellWidth() const;
    point_type GetOrigin() const;

    index_tuple GetIndexOrigin() const;
    size_tuple GetDimensions() const;

    index_tuple GetGhostWidth(int tag = VERTEX) const;

    index_box_type GetIndexBox(int tag) const;

    point_type point(entity_id_type s) const { return local_coordinates(s.x, s.y, s.z, s.w & 0b111); };

    point_type local_coordinates(index_type x, index_type y, index_type z, Real const r[3]) const {
        return GetChart().local_coordinates(x, y, z, r);
    }

    point_type local_coordinates(index_type x, index_type y, index_type z, int tag = 0) const {
        return GetChart().local_coordinates(x, y, z, EntityIdCoder::m_id_to_coordinates_shift_[tag & 0b111]);
    }
    point_type local_coordinates(entity_id_type s, Real const *pr) const {
        Real r[3];

        r[0] = pr[0] + EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][0];
        r[1] = pr[1] + EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][1];
        r[2] = pr[2] + EntityIdCoder::m_id_to_coordinates_shift_[s.w & 0b111][2];

        return GetChart().local_coordinates(s.x, s.y, s.z, r);
    }

    template <typename... Args>
    point_type global_coordinates(Args &&... args) const {
        return map(local_coordinates(std::forward<Args>(args)...));
    }

    ZSFC<NDIMS> GetSpaceFillingCurve(int iform, int nsub = 0) const {
        return ZSFC<NDIMS>{GetIndexBox(EntityIdCoder::m_sub_index_to_id_[iform][nsub])};
    }

    template <typename U>
    using array_type = Array<U, ZSFC<NDIMS>>;

    template <typename U, int IFORM, int... N>
    using data_type = nTuple<Array<U, ZSFC<NDIMS>>, (IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3, N...>;

    template <int IFORM, typename U, int... N>
    void initialize_data(nTuple<array_type<U>, N...> *d) const {
        traits::foreach (*d, [&](auto &a, int n0, auto &&... idx) { a.reset(GetSpaceFillingCurve(IFORM, n0)); });
    };

   public:
    size_type GetNumberOfEntity(int IFORM = VERTEX) const {
        index_box_type m_index_box_ = GetBlock().GetIndexBox();
        return calculus::reduction<tags::multiplication>(std::get<1>(m_index_box_) - std::get<0>(m_index_box_)) *
               ((IFORM == VERTEX || IFORM == VOLUME) ? 1 : 3);
    }
};
}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_STRUCTUREDMESH_H
