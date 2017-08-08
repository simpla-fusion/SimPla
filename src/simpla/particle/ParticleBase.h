//
// Created by salmon on 16-6-13.
//

#ifndef SIMPLA_PARTICLEBASE_H
#define SIMPLA_PARTICLEBASE_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/engine/Attribute.h"
#include "simpla/engine/SPObject.h"
#include "simpla/utilities/Range.h"
#include "spParticle.h"
namespace simpla {

class ParticleBase {
    SP_OBJECT_BASE(ParticleBase);

   public:
    ParticleBase();
    ~ParticleBase();
    ParticleBase(ParticleBase const &) = delete;
    ParticleBase(ParticleBase &&) = delete;
    ParticleBase &operator=(ParticleBase const &) = delete;
    ParticleBase &operator=(ParticleBase &&) = delete;

    void DoInitialize();

    struct Bucket {
        int m_depth_ = 0;
        id_type *m_tag_ = nullptr;
        void **m_data_ = nullptr;
    };

    Bucket *FindBucket(id_type s);
    Bucket *GetBucket(id_type s);
    Bucket const *GetBucket(id_type s) const;

    void Sort();
    void DeepSort();

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace simpla{namespace particle{
#endif  // SIMPLA_PARTICLEBASE_H
