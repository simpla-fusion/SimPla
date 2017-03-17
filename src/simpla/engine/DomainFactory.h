//
// Created by salmon on 17-3-17.
//

#ifndef SIMPLA_DOMAINFACTORY_H
#define SIMPLA_DOMAINFACTORY_H

#include <simpla/data/all.h>
#include <simpla/design_pattern/SingletonHolder.h>
#include <memory>
namespace simpla {
namespace engine {
class MeshView;
class Worker;
struct DomainFactory {
   public:
    DomainFactory();
    ~DomainFactory();
    bool RegisterMeshCreator(
        std::string const &k,
        std::function<std::shared_ptr<MeshView>(std::shared_ptr<data::DataTable> const &)> const &);

    template <typename U>
    bool RegisterMeshCreator(std::string const &k) {
        RegisterMeshCreator(k, [&](std::shared_ptr<data::DataTable> const &t) { return std::make_shared<U>(t); });
    }

    bool RegisterWorkerCreator(
        std::string const &k, std::function<std::shared_ptr<Worker>(std::shared_ptr<data::DataTable> const &)> const &);

    template <typename U>
    bool RegisterWorkerCreator(std::string const &k) {
        RegisterWorkerCreator(k, [&](std::shared_ptr<data::DataTable> const &t) { return std::make_shared<U>(t); });
    }
    std::shared_ptr<MeshView> CreateMesh(std::shared_ptr<data::DataEntity> const &p);
    std::shared_ptr<Worker> CreateWorker(std::shared_ptr<MeshView> const &m, std::shared_ptr<data::DataEntity> const &);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
#define GLOBAL_DOMAIN_FACTORY SingletonHolder<DomainFactory>::instance()

}  // namespace engine
}  // namespace simpla{
#endif  // SIMPLA_DOMAINFACTORY_H
