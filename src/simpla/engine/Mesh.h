//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/concept/Printable.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "SPObject.h"
namespace simpla {
namespace engine {
class MeshBlock;
class Patch;
class IdRange;
/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class Mesh : public concept::Configurable {
    SP_OBJECT_BASE(Mesh);

   public:
    Mesh();
    Mesh(Mesh const &);
    virtual ~Mesh();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    virtual Mesh *Clone() const = 0;

    virtual void Register(AttributeGroup *);
    virtual void Deregister(AttributeGroup *);
    virtual void Push(Patch const &);
    virtual void Pop(Patch *) const;

    id_type GetBlockId() const;
    //    void SetBlock(const std::shared_ptr<MeshBlock> &);
    std::shared_ptr<MeshBlock> const &GetBlock() const;
    std::shared_ptr<IdRange> const &GetRange(int iform) const;

    virtual void Initialize();
    virtual void Finalize();

    static bool RegisterCreator(std::string const &k, std::function<Mesh *()> const &);
    static Mesh *Create(std::shared_ptr<data::DataTable> const &);

    template <typename U>
    static bool RegisterCreator(std::string const &k) {
        return RegisterCreator(k, [&]() -> Mesh * { return new U; });
    }

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename...>
class MeshView;
//    template <typename U>
//    std::shared_ptr<data::DataBlockWrapper<U>> CreateDataBlock(int IFORM, int DOF) const;
//    template <typename... Args>
//    Range<EntityId> range(Args &&... args) const {
//        if (m_mesh_block_ != nullptr) {
//            return m_mesh_block_->range(std::forward<Args>(args)...);
//        } else {
//            return Range<EntityId>();
//        }
//    }
//    size_type size(int IFORM = VERTEX) const { return m_mesh_block_->number_of_entities(IFORM); }
//    template <typename... Args>
//    auto Hash(Args &&... args) const {
//        return m_mesh_block_->Hash(std::forward<Args>(args)...);
//    }
//    template <typename... Args>
//    auto pack(Args &&... args) const {
//        return m_mesh_block_->pack(std::forward<Args>(args)...);
//    }
//    point_type dx() const {
//        if (m_mesh_block_ != nullptr) {
//            return m_mesh_block_->dx();
//        } else {
//            return point_type{1, 1, 1};
//        }
//    }
//    template <typename... Args>
//    decltype(auto) point_global_to_local(Args &&... args) const {
//        return m_mesh_block_->point_global_to_local(std::forward<Args>(args)...);
//    }
//    template <typename... Args>
//    decltype(auto) point(Args &&... args) const {
//        return m_mesh_block_->point(std::forward<Args>(args)...);
//    }
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_GEOMETRY_H
