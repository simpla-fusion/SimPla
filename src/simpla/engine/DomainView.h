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
class MeshBlock;
class AttributeView;
class DataBlock;
class Patch;

class DomainView : public concept::Printable, public SPObject {
   public:
    DomainView();
    virtual ~DomainView();
    std::ostream &Print(std::ostream &os, int indent) const final;
    id_type current_block_id() const;
    void Dispatch(std::shared_ptr<Patch> d);

    virtual bool Update();
    void Evaluate();

    //    Manager const *GetManager(Manager *) const;
    //    void SetManager(Manager *m = nullptr);

    MeshView &GetMesh() const;

    template <typename U, typename... Args>
    U &SetMesh(Args &&... args) {
        auto res = std::make_shared<U>(std::forward<Args>(args)...);
        SetMesh(std::dynamic_pointer_cast<MeshView>(res));
        Attach(dynamic_cast<AttributeViewBundle *>(res.get()));
        return *res;
    };

   private:
    void SetMesh(std::shared_ptr<MeshView> const &m);

   public:
    id_type GetMeshBlockId() const;
    std::shared_ptr<MeshBlock> &GetMeshBlock() const;
    std::shared_ptr<DataBlock> &GetDataBlock(id_type) const;

    std::pair<Worker &, bool> AddWorker(std::shared_ptr<Worker> const &w, int pos = -1);
    void RemoveWorker(std::shared_ptr<Worker> const &w);
    template <typename U>
    U &AddWorker(int pos = -1, ENABLE_IF((std::is_base_of<Worker, U>::value))) {
        auto res = AddWorker(std::make_shared<U>(), pos);
        return dynamic_cast<U &>(res.first);
    };

    void Attach(AttributeViewBundle *);
    void Detach(AttributeViewBundle *p = nullptr);
    void Notify();

    void Register(AttributeDict &);

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
