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
#include <simpla/engine/SPObject.h>
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
        int size = 0;
        int tag[SP_MAX_PARTICLE_IN_CELL];
        Real data[][SP_MAX_PARTICLE_IN_CELL];
    };

    auto at(EntityId s) const { return std::make_tuple(m_data_->lower_bound(s), m_data_->upper_bound(s)); }
    auto at(EntityId s) { return std::make_tuple(m_data_->lower_bound(s), m_data_->upper_bound(s)); }

    auto operator[](EntityId s) const { return at(s); }
    auto operator[](EntityId s) { return at(s); }
    void Unpack(engine::DataPack &&d) override {
        if (d != nullptr) { Click(); }
        DoUpdate();
    }

    std::shared_ptr<data::DataBlock> Pack() override {
        auto res = std::make_shared<data::DataBlock>();
        DoTearDown();
        return res;
    }
    std::shared_ptr<bucket_s> pop_bucket();
    void push_bucket(std::shared_ptr<bucket_s> const&);

    template <typename TFun>
    size_type Advance(int tag, bucket_s const* src, bucket_s* dest, TFun const&);

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

void scan(int num, int const* in, int* out) {}

template <typename T>
void filter(int num, int dest, int const* tag, int* o_idx) {
    int idx[num];
    for (int i = 0; i < num; ++i) { idx[i] = tag[i] == dest ? 1 : 0; }
    scan(num, idx, o_idx);
}
template <typename TFun>
size_type ParticleBase::Advance(int tag, bucket_s const* src, bucket_s* dest, TFun const& fun) {
    int num = src->size;
    int o_idx[num];
    filter(num, tag, src->tag, o_idx);
    int dest_min = dest->size;
    int dest_max = o_idx[num - 1] - (SP_MAX_PARTICLE_IN_CELL - dest->size);
    for (int i = 0; i < num; ++i) {
        if (src->tag[i] == tag && o_idx[i] + dest_min < SP_MAX_PARTICLE_IN_CELL && o_idx[i] < dest_max)
            fun(src->data, i, dest->data, o_idx[i] + dest_min);
    }
};
}  // namespace simpla{namespace particle{
#endif  // SIMPLA_PARTICLEBASE_H
