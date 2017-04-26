//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::engine;

struct SMesh : public StructuredMesh {
   public:
    SP_OBJECT_HEAD(SMesh, StructuredMesh)

     explicit SMesh(std::shared_ptr<engine::Chart> c = nullptr) : StructuredMesh(c){};
    ~SMesh() override{};

    SP_DEFAULT_CONSTRUCT(SMesh)
    DECLARE_REGISTER_NAME("SMesh");
    void InitializeData(Real time_now) override;

   private:
    Field<this_type, Real, VERTEX, 3> m_vertics_{this, "name"_ = "vertics", "COORDINATES"_};
    Field<this_type, Real, VOLUME, 9> m_volume_{this, "name"_ = "volume"};
    Field<this_type, Real, VOLUME, 9> m_dual_volume_{this, "name"_ = "dual_volume"};
    Field<this_type, Real, VOLUME, 9> m_inv_volume_{this, "name"_ = "inv_volume"};
    Field<this_type, Real, VOLUME, 9> m_inv_dual_volume_{this, "name"_ = "inv_dual_volume"};
    typedef EntityIdCoder M;

   public:
    point_type point(index_type i, index_type j, index_type k) const override {
        return point_type{m_vertics_[0](i, j, k), m_vertics_[1](i, j, k), m_vertics_[2](i, j, k)};
    };
    point_type point(EntityId s) const override { return StructuredMesh::point(s); }

    Real volume(EntityId s) const override { return m_volume_[s.w & 7](s.x, s.y, s.z); }
    Real dual_volume(EntityId s) const override { return m_volume_[s.w & 7](s.x, s.y, s.z); }
    Real inv_volume(EntityId s) const override { return m_volume_[s.w & 7](s.x, s.y, s.z); }
    Real inv_dual_volume(EntityId s) const override { return m_volume_[s.w & 7](s.x, s.y, s.z); }

   protected:
    auto &GetVertics() const { return m_vertics_; };
    auto &GetVolume() const { return m_volume_; };
    auto &GetDualVolume() const { return m_dual_volume_; };
    auto &GetInvVolume() const { return m_inv_volume_; };
    auto &GetInvDualVolume() const { return m_inv_dual_volume_; };

    auto &GetVertics() { return m_vertics_; };
    auto &GetVolume() { return m_volume_; };
    auto &GetDualVolume() { return m_dual_volume_; };
    auto &GetInvVolume() { return m_inv_volume_; };
    auto &GetInvDualVolume() { return m_inv_dual_volume_; };
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
