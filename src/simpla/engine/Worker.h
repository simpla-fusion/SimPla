//
// Created by salmon on 17-4-5.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <memory>
#include "Attribute.h"
#include "simpla/data/all.h"

namespace simpla {
namespace engine {
class Mesh;
class Patch;
class AttributeGroup;
/**
* @brief
*/
class Worker : public data::Serializable, public data::EnableCreateFromDataTable<Worker> {
    SP_OBJECT_BASE(engine::Worker)
   public:
    Worker();
    Worker(Worker const &) = delete;
    virtual ~Worker();

    //    virtual Worker *Clone() const;

    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable>);

    virtual void Register(AttributeGroup *);
    virtual void Deregister(AttributeGroup *);

    virtual void Push(std::shared_ptr<Patch>);
    virtual std::shared_ptr<Patch> Pop();

    virtual void Initialize();
    virtual void SetUp();

    virtual void TearDown();
    virtual void Finalize();

    virtual Mesh *GetMesh() = 0;
    virtual Mesh const *GetMesh() const = 0;

    virtual void InitializeCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void Advance(Real time_now, Real dt);
};
}
}
#endif  // SIMPLA_WORKER_H
