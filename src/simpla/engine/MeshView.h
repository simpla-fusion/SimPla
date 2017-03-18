//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/concept/Printable.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <simpla/geometry/GeoObject.h>
#include "AttributeView.h"
#include "SPObject.h"
namespace simpla {
namespace engine {
class MeshBlock;
/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class MeshView : public AttributeViewBundle {
   public:
    SP_OBJECT_BASE(MeshView);
    MeshView(std::shared_ptr<data::DataEntity> const &t, std::shared_ptr<geometry::GeoObject> const &obj = nullptr);
    virtual ~MeshView();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;
    id_type GetMeshBlockId() const;
    std::shared_ptr<MeshBlock> const &GetMeshBlock() const;
    void SetMeshBlock(std::shared_ptr<MeshBlock> const &);
    std::shared_ptr<geometry::GeoObject> GetGeoObject() const;

    virtual void PushPatch(std::shared_ptr<Patch> const &p);
    virtual std::shared_ptr<Patch> PopPatch() const;
    virtual bool Update();
    virtual void Initialize();
    Real GetDt() const;

    size_type size(int IFORM = VERTEX) const { return 0; }
    size_tuple dimensions() const { return size_tuple{}; };
    //    template <typename... Args>
    //    Range<MeshEntityId> range(Args &&... args) const {
    //        if (m_mesh_block_ != nullptr) {
    //            return m_mesh_block_->range(std::forward<Args>(args)...);
    //        } else {
    //            return Range<MeshEntityId>();
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

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
struct MeshViewFactory {
   public:
    MeshViewFactory();
    ~MeshViewFactory();

    bool RegisterCreator(
        std::string const &k,
        std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataEntity> const &,
                                                std::shared_ptr<geometry::GeoObject> const &)> const &);

    template <typename U>
    bool RegisterCreator(std::string const &k) {
        RegisterCreator(k, [&](std::shared_ptr<data::DataEntity> const &t,
                               std::shared_ptr<geometry::GeoObject> const &g) { return std::make_shared<U>(t, g); });
    }

    std::shared_ptr<MeshView> Create(std::shared_ptr<data::DataEntity> const &p,
                                     std::shared_ptr<geometry::GeoObject> const &g = nullptr);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define GLOBAL_MESHVIEW_FACTORY SingletonHolder<MeshViewFactory>::instance()

template <typename M>
class MeshAdapter : public MeshView, public M {
   public:
    MeshAdapter(){};
    template <typename... Args>
    explicit MeshAdapter(Args &&... args) : MeshView(), M(std::forward<Args>(args)...) {}
    ~MeshAdapter() {}

    void Initialize() final { M::Initialize(); };
};
}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_GEOMETRY_H
