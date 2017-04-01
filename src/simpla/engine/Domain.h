//
// Created by salmon on 17-2-10.
//

#ifndef SIMPLA_DOMAIN_H
#define SIMPLA_DOMAIN_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <memory>
#include "AttributeView.h"
#include "simpla/geometry/GeoObject.h"
#include "simpla/mpl/macro.h"

namespace simpla {
namespace engine {
class AttributeView;
class MeshView;
class MeshBlock;
class DataBlock;
class DomainView;
class Worker;

class Domain : public concept::Configurable {
   public:
    Domain();
    ~Domain();

    void RegisterMeshFactory(
        std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataTable> const &,
                                                std::shared_ptr<geometry::GeoObject> const &)> const &);
    template <typename U>
    void RegisterMeshFactory(ENABLE_IF((std::is_base_of<MeshView, U>::value))) {
        RegisterMeshFactory([](std::shared_ptr<data::DataTable> const &t,
                               std::shared_ptr<geometry::GeoObject> const &g) -> std::shared_ptr<MeshView> {
            return std::make_shared<U>(t, g);
        });
    };

    std::shared_ptr<DomainView> CreateView();

    void SetMeshView(std::shared_ptr<MeshView> const &m);
    std::shared_ptr<MeshView> &GetMeshView() const;
    std::shared_ptr<MeshView> CreateMeshView();
    void SetGeoObject(std::shared_ptr<geometry::GeoObject> const &);
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

    std::pair<std::shared_ptr<Worker>, bool> AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    void RemoveWorker(std::shared_ptr<Worker> const &w);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

class DomainView {
   public:
    DomainView(std::shared_ptr<MeshView> const &m = nullptr, std::shared_ptr<geometry::GeoObject> const &g = nullptr);
    virtual ~DomainView();
    virtual void Initialize();
    virtual void Finalize();

    std::shared_ptr<MeshView> const &GetMeshView() const;
    std::shared_ptr<geometry::GeoObject> const &GetGeoObject() const;

    virtual void PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataTable> const &);
    virtual void PushData(std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> const &);
    virtual std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataTable>> PopData();

    virtual void Run(Real dt);

    std::pair<std::shared_ptr<Worker>, bool> AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    void RemoveWorker(std::shared_ptr<Worker> const &w);

    void Attach(AttributeViewBundle *);
    void Detach(AttributeViewBundle *p = nullptr);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine {
}  // namespace simpla {
#endif  // SIMPLA_DOMAIN_H
