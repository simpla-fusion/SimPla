//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <memory>
#include "simpla/concept/Configurable.h"
#include "simpla/data/all.h"
namespace simpla {

namespace engine {
class Mesh;

class AttributeBundle;
/**
* @brief
*/
class Worker : public concept::Configurable {
   public:
    Worker();
    Worker(Worker const &);
    virtual ~Worker();
    virtual void swap(Worker &);
    virtual Worker *Clone() const;

    virtual void Register(AttributeBundle *);
    virtual void Deregister(AttributeBundle *);

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
