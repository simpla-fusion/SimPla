//
// Created by salmon on 17-2-12.
//

#ifndef SIMPLA_DOMAINVIEW_H
#define SIMPLA_DOMAINVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/mpl/macro.h>
#include <memory>
#include <set>
#include "AttributeView.h"

namespace simpla {
namespace engine {
class Domain;
class Worker;
class MeshView;
class AttributeView;
class DataBlock;
class Patch;

class DomainView : public concept::Printable {
   public:
    DomainView();
    virtual ~DomainView();
    std::ostream &Print(std::ostream &os, int indent) const final;
    id_type current_block_id() const;
    void Dispatch(std::shared_ptr<Patch> const &d);
    bool isUpdated() const;
    void Update();
    void Evaluate();

    void SetMesh(std::shared_ptr<MeshView> const &m);
    std::shared_ptr<MeshView> const &GetMesh() const;
    void AppendWorker(std::shared_ptr<Worker> const &w);
    void PrependWorker(std::shared_ptr<Worker> const &w);
    void RemoveWorker(std::shared_ptr<Worker> const &w);

    template <typename U>
    void SetMesh(std::shared_ptr<U> const &m = nullptr, ENABLE_IF((std::is_base_of<MeshView, U>::value))) {
        SetMesh(std::dynamic_pointer_cast<MeshView>(std::make_shared<U>()));
    };

    template <typename U>
    void AppendWorker(std::shared_ptr<U> const &w = nullptr, ENABLE_IF((std::is_base_of<Worker, U>::value))) {
        AppendWorker(std::dynamic_pointer_cast<Worker>(std::make_shared<U>()));
    };

    template <typename U>
    void PrependWorker(std::shared_ptr<U> const &w = nullptr, ENABLE_IF((std::is_base_of<Worker, U>::value))) {
        PrependWorker(std::dynamic_pointer_cast<Worker>(std::make_shared<U>()));
    };

    std::shared_ptr<MeshBlock> const &mesh_block() const;
    std::shared_ptr<DataBlock> data_block(id_type) const;
    void data_block(id_type, std::shared_ptr<DataBlock> const &);
    void UpdateAttributeDict();
    std::tuple<std::string, std::type_index, int, int> GetAttributeDict(id_type) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename>
struct DomainViewAdapter : public DomainView {};
}  // namespace engine {
}  // namespace simpla {

#endif  // SIMPLA_DOMAINVIEW_H
