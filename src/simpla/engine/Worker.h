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
class MeshBase;
class Patch;
class AttributeGroup;
/**
* @brief
*/
class Worker : public data::Serializable, public data::EnableCreateFromDataTable<Worker, std::shared_ptr<MeshBase>> {
    SP_OBJECT_BASE(Worker)
   public:
    explicit Worker(std::shared_ptr<MeshBase> m = nullptr);
    ~Worker() override;

    SP_DEFAULT_CONSTRUCT(Worker);
    DECLARE_REGISTER_NAME("Worker")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    std::shared_ptr<MeshBase> GetMesh() { return m_mesh_; }
    std::shared_ptr<MeshBase> const GetMesh() const { return m_mesh_; }

    virtual void Push(Patch*);
    virtual void Pop(Patch*);

    virtual void Initialize();
    virtual void SetUp();
    virtual void TearDown();
    virtual void Finalize();

    virtual void InitializeCondition(Real time_now);
    virtual void BoundaryCondition(Real time_now, Real dt);
    virtual void Advance(Real time_now, Real dt);

   private:
    std::shared_ptr<MeshBase> m_mesh_;
};

#define WORKER_HEAD(_WORKER_NAME_)                                                                                     \
   public:                                                                                                             \
    explicit _WORKER_NAME_(std::shared_ptr<engine::MeshBase> m = nullptr)                                              \
        : engine::Worker((m != nullptr) ? m                                                                            \
                                        : std::dynamic_pointer_cast<engine::MeshBase>(std::make_shared<mesh_type>())), \
          m_mesh_(std::dynamic_pointer_cast<mesh_type>(engine::Worker::GetMesh()).get()) {}                            \
    ~_WORKER_NAME_() override = default;                                                                               \
    SP_DEFAULT_CONSTRUCT(_WORKER_NAME_);                                                                               \
    DECLARE_REGISTER_NAME(std::string(__STRING(_WORKER_NAME_)) + "<" + mesh_type::ClassName() + ">")                   \
    mesh_type* m_mesh_;                                                                                                \
    template <int IFORM, int DOF = 1>                                                                                  \
    using field_type = Field<mesh_type, typename mesh_type::scalar_type, IFORM, DOF>;
}
}
#endif  // SIMPLA_WORKER_H
