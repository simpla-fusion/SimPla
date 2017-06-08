//
// Created by salmon on 16-6-13.
//

#ifndef SIMPLA_PARTICLEBASE_H
#define SIMPLA_PARTICLEBASE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/Attribute.h>
#include <simpla/utilities/EntityId.h>
#include <simpla/utilities/Range.h>
#include <simpla/utilities/nTuple.h>
#include <simpla/utilities/sp_def.h>
#include "spParticle.h"
namespace simpla {

class ParticleBase : public engine::Attribute {
    SP_OBJECT_HEAD(ParticleBase, ParticleBase);

   public:
    template <typename... Args>
    explicit ParticleBase(int DOF, Args&&... args)
        : engine::Attribute(FIBER, DOF, typeid(Real), std::forward<Args>(args)...){};
    ~ParticleBase() override = default;

    static constexpr int SP_MAX_PARTICLE_IN_CELL = 256;

    struct bucket_s {
        size_type size = 0;
        int tag[SP_MAX_PARTICLE_IN_CELL];
        Real d[][SP_MAX_PARTICLE_IN_CELL];
    };

    auto at(EntityId s) const { return std::make_tuple(m_data_->lower_bound(s), m_data_->upper_bound(s)); }
    auto at(EntityId s) { return std::make_tuple(m_data_->lower_bound(s), m_data_->upper_bound(s)); }

    auto operator[](EntityId s) const { return at(s); }
    auto operator[](EntityId s) { return at(s); }
    void Push(const std::shared_ptr<data::DataBlock>& d, const EntityRange& r) override {
        if (d != nullptr) { Click(); }
        DoUpdate();
    }

    std::shared_ptr<data::DataBlock> Pop() override {
        auto res = std::make_shared<data::DataBlock>();
        DoTearDown();
        return res;
    }
    std::shared_ptr<bucket_s> pop_bucket();
    void push_bucket(std::shared_ptr<bucket_s> const&);

    template <typename TFun>
    void Advance(int tag, bucket_s const* src, bucket_s* dest, Real const* E[3], Real const* B[3], Real* J[3]);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
template <typename TFun>
void ParticleBase::Advance(int tag, bucket_s const* src, bucket_s* dest, TFun const& fun, Real const* E[3],
                           Real const* B[3], Real* J[3]){

};
}  // namespace simpla{namespace particle{
#endif  // SIMPLA_PARTICLEBASE_H
