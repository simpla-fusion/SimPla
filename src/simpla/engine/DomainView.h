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
    virtual std::ostream &Print(std::ostream &os, int indent) const;

    id_type current_block_id() const;
    void Dispatch(Patch const &d);
    bool isUpdated() const;
    virtual void Update();
    virtual void Evaluate();

    void SetMesh(std::shared_ptr<MeshView> const &m);
    std::shared_ptr<MeshView> const &GetMesh() const;

    void AppendWorker(std::shared_ptr<Worker> w);
    void PrependWorker(std::shared_ptr<Worker> w);
    void RemoveWorker(std::shared_ptr<Worker> w);

    template <typename U>
    void SetMesh(std::shared_ptr<U> const &m = nullptr, ENABLE_IF((std::is_base_of<MeshView, U>::value))) {
        SetMesh(std::dynamic_pointer_cast<MeshView>(std::make_shared<U>()));
    };
    template <typename U>
    void AppendWorker(std::shared_ptr<U> w = nullptr, ENABLE_IF((std::is_base_of<Worker, U>::value))) {
        AppendWorker(std::dynamic_pointer_cast<Worker>(std::make_shared<U>()));
    };
    template <typename U>
    void PrependWorker(std::shared_ptr<U> w = nullptr, ENABLE_IF((std::is_base_of<Worker, U>::value))) {
        PrependWorker(std::dynamic_pointer_cast<Worker>(std::make_shared<U>()));
    };
    std::shared_ptr<MeshBlock> const &mesh_block() const;
    std::shared_ptr<DataBlock> data_block(id_type) const;
    void data_block(id_type, std::shared_ptr<DataBlock>);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_DOMAINVIEW_H
