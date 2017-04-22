//
// Created by salmon on 17-4-22.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H

#include <simpla/algebra/nTupleExt.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/sp_def.h>

namespace simpla {
namespace mesh {
template <typename BMesh>
class EBMesh : public BMesh {
   private:
    typedef BMesh base_type;
    typedef EBMesh<base_type> this_type;

   public:
    bool isA(std::type_info const &info) const override { return typeid(this_type) == info || base_type::isA(info); }
    std::type_info const &GetTypeInfo() const override { return typeid(this_type); }
    std::string GetClassName() const override { return "EBMesh<" + BMesh::ClassName() + ">"; }
    static std::string ClassName() { return "EBMesh<" + BMesh::ClassName() + ">"; }

   public:
    template <typename... Args>
    explicit EBMesh(Args &&... args) : BMesh(std::forward<Args>(args)...) {}
    ~EBMesh() override = default;

    SP_DEFAULT_CONSTRUCT(EBMesh)

    //    using BMesh::point;
    //
    //    Range<EntityId> GetRange(int iform) const override {}
    //
    //    Real volume(EntityId s) const override { return 0.0; }
    //    Real dual_volume(EntityId s) const override { return 0.0; }
    //    Real inv_volume(EntityId s) const override { return 0.0; }
    //    Real inv_dual_volume(EntityId s) const override { return 0.0; }
};
}
}

#endif  // SIMPLA_EBMESH_H
