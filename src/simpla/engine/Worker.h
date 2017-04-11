//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <memory>
#include "Attribute.h"
#include "simpla/concept/Serializable.h"
#include "simpla/data/all.h"
namespace simpla {

namespace engine {
class Mesh;
class Patch;
class AttributeGroup;
/**
* @brief
*/
class Worker : public concept::Serializable<Worker>, public AttributeGroup {
    SP_OBJECT_BASE(engine::Worker)
   public:
    Worker();
    Worker(Worker const &);
    virtual ~Worker();
    virtual void swap(Worker &);
    virtual Worker *Clone() const;

    virtual void Register(AttributeGroup *);
    virtual void Deregister(AttributeGroup *);
    virtual void Push(Patch &p);
    virtual void Pop(Patch *);

    virtual void Initialize(Real time_now = 0);
    virtual void Advance(Real time = 0, Real dt = 0);
    virtual void Finalize();

    void SetMesh(std::shared_ptr<Mesh> const &);
    std::shared_ptr<Mesh> const &GetMesh() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_WORKER_H
