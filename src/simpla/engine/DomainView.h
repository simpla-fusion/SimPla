//
// Created by salmon on 17-2-12.
//

#ifndef SIMPLA_DOMAINVIEW_H
#define SIMPLA_DOMAINVIEW_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Printable.h>
#include <simpla/data/DataBlock.h>
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
class Patch;

class DomainView : public concept::Printable, public SPObject {
   public:
    DomainView(std::shared_ptr<data::DataTable> const &p = nullptr);
    virtual ~DomainView();
    std::ostream &Print(std::ostream &os, int indent) const final;

    id_type current_block_id() const;
    void Dispatch(std::shared_ptr<Patch> d);

    virtual bool Update();
    void Run(Real dt);

    //    Manager const *GetManager(Manager *) const;
    //    void SetManager(Manager *m = nullptr);

    std::shared_ptr<MeshView> GetMesh() const;

    //    template <typename U, typename... Args>
    //    U &SetMesh(Args &&... args) {
    //        auto res = std::make_shared<U>(std::forward<Args>(args)...);
    //        SetMesh(std::dynamic_pointer_cast<MeshView>(res));
    //        Attach(dynamic_cast<AttributeViewBundle *>(res.get()));
    //        return *res;
    //    };

    std::shared_ptr<MeshView> SetMesh(std::shared_ptr<MeshView> const &m);

    id_type GetMeshBlockId() const;
    std::shared_ptr<MeshBlock> GetMeshBlock() const;
    std::shared_ptr<data::DataBlock> GetDataBlock(id_type) const;

    void RemoveWorker(std::shared_ptr<Worker> const &w);

    std::pair<std::shared_ptr<Worker>, bool> AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    template <typename U, typename... Args>
    U &AddWorker(Args &&... args) {
        auto res = AddWorker(std::make_shared<U>(std::forward<Args>(args)...));
        //        if (!res.second) { res.first = ; }
        return *std::dynamic_pointer_cast<U>(res.first);
    };
    void Attach(AttributeViewBundle *);
    void Detach(AttributeViewBundle *p = nullptr);
    void Notify();
    void Initialize();
    //    Range<id_type> const &select(int GetIFORM, int GetTag);
    //    Range<id_type> const &select(int GetIFORM, std::string const &GetTag);
    //    Range<id_type> const &interface(int GetIFORM, const std::string &tag_in, const std::string &tag_out =
    //    "VACUUM");
    //    Range<id_type> const &interface(int GetIFORM, int tag_in, int tag_out);
    //    Range<id_type> const &select(int GetIFORM, int GetTag) const;
    //    Range<id_type> const &interface(int GetIFORM, int tag_in, int tag_out ) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename>
struct DomainViewAdapter : public DomainView {};
}  // namespace engine {
}  // namespace simpla {

#endif  // SIMPLA_DOMAINVIEW_H
