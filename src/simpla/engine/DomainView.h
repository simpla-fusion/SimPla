//
// Created by salmon on 17-2-12.
//

#ifndef SIMPLA_DOMAINVIEW_H
#define SIMPLA_DOMAINVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/StateCounter.h>
#include <simpla/mpl/macro.h>
#include <memory>
#include <set>
#include "AttributeView.h"

namespace simpla {
namespace engine {
class Domain;
class Worker;
class MeshView;
class MeshBlock;
class AttributeView;
class DataBlock;
class Patch;

class DomainView : public concept::Printable, public concept::StateCounter {
   public:
    DomainView();
    virtual ~DomainView();
    std::ostream &Print(std::ostream &os, int indent) const final;
    id_type current_block_id() const;
    void Dispatch(std::shared_ptr<Patch> d);
    virtual bool isUpdated() const;
    virtual void Update();
    void Evaluate();

    //    Manager const *GetManager(Manager *) const;
    //    void SetManager(Manager *m = nullptr);

    template <typename U, typename... Args>
    U &CreateMesh(Args &&... args) {
        auto res = std::make_shared<U>(std::forward<Args>(args)...);
        SetMesh(std::dynamic_pointer_cast<MeshView>(res));
        Attach(static_cast<AttributeViewBundle *>(res.get()));
        return *res;
    };
    void SetMesh(std::shared_ptr<MeshView> const &m);
    MeshView &GetMesh() const;

    id_type GetMeshBlockId() const;
    std::shared_ptr<MeshBlock> &GetMeshBlock() const;
    std::shared_ptr<DataBlock> &GetDataBlock(id_type) const;

    void AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    void RemoveWorker(std::shared_ptr<Worker> const &w);

    void Attach(AttributeViewBundle *);
    void Detach(AttributeViewBundle *p = nullptr);
    void Notify();

    void RegisterAttribute(AttributeDict *);
    //    void UpdateAttributeDict();
    std::map<id_type, std::shared_ptr<engine::AttributeDesc>> const &GetAttributeDict() const;
    data::DataTable const &attr_db(id_type) const;
    data::DataTable &attr_db(id_type);
    //    Range<id_type> const &select(int iform, int tag);
    //    Range<id_type> const &select(int iform, std::string const &tag);
    //    Range<id_type> const &interface(int iform, const std::string &tag_in, const std::string &tag_out = "VACUUM");
    //    Range<id_type> const &interface(int iform, int tag_in, int tag_out);
    //    Range<id_type> const &select(int iform, int tag) const;
    //    Range<id_type> const &interface(int iform, int tag_in, int tag_out ) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename>
struct DomainViewAdapter : public DomainView {};
}  // namespace engine {
}  // namespace simpla {

#endif  // SIMPLA_DOMAINVIEW_H
