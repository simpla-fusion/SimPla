//
// Created by salmon on 17-6-8.
//

#ifndef SIMPLA_PARTICLE_H
#define SIMPLA_PARTICLE_H

#include <simpla/algebra/Field.h>
#include "simpla/SIMPLA_config.h"

#include "simpla/engine/Attribute.h"

namespace simpla {
namespace engine {
struct MeshBase;
}
class ParticleBase {
   public:
    explicit ParticleBase(engine::MeshBase const* m = nullptr, int DOF = 3);
    virtual ~ParticleBase();
    SP_DEFAULT_CONSTRUCT(ParticleBase);
    virtual std::shared_ptr<data::DataTable> GetProperties() const { return nullptr; };
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(const std::shared_ptr<data::DataTable>& t);

    virtual void PushData(data::DataBlock* dblk);
    virtual void PopData(data::DataBlock* dblk);

    int GetNumberOfAttributes() const;
    size_type GetMaxSize() const;

    struct Bucket {
        std::shared_ptr<Bucket> next = nullptr;
        size_type count = 0;
        int* tag = nullptr;
        Real** data = nullptr;
    };
    std::shared_ptr<Bucket> GetBucket(id_type s = NULL_ID);
    std::shared_ptr<Bucket> AddBucket(id_type s, size_type num);
    void RemoveBucket(id_type s);
    std::shared_ptr<Bucket> GetBucket(id_type s = NULL_ID) const;

    virtual void DoInitialize();
    void InitialLoad(int const* rnd_type = nullptr, size_type rnd_offset = 0);

    size_type Count(id_type s = NULL_ID) const;
    void Sort();
    void DeepSort();

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template <typename TM>
class Particle : public ParticleBase, public engine::Attribute, public data::Serializable {
    SP_OBJECT_HEAD(Particle<TM>, engine::Attribute);

   public:
    typedef TM mesh_type;
    static constexpr int iform = FIBER;
    static constexpr int ndims = 3;

   private:
    mesh_type const* m_host_ = nullptr;

   public:
    template <typename... Args>
    Particle(mesh_type* grp, int DOF, Args&&... args)
        : ParticleBase(grp->GetMesh(), DOF),
          base_type(grp->GetMesh(), FIBER, DOF, typeid(Real),
                    std::make_shared<data::DataTable>(std::forward<Args>(args)...)),
          m_host_(grp) {}

    ~Particle() override = default;

    SP_DEFAULT_CONSTRUCT(Particle);

    std::shared_ptr<data::DataTable> GetProperties() const override { return engine::Attribute::db(); }

    std::shared_ptr<data::DataTable> Serialize() const override { return ParticleBase::Serialize(); }

    void Deserialize(const std::shared_ptr<data::DataTable>& t) override { ParticleBase::Deserialize(t); }

    void DoInitialize() override {
        if (base_type::isNull()) {
            ParticleBase::DoInitialize();
        } else {
            ParticleBase::PushData(GetDataBlock());
        }
    }
    void DoFinalize() override { ParticleBase::PopData(GetDataBlock()); }

};  // class Particle
}  // namespace simpla{

#endif  // SIMPLA_PARTICLE_H
