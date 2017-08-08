//
// Created by salmon on 17-6-8.
//
#include "ParticleBase.h"
namespace simpla {
struct ParticleBase::pimpl_s {
    struct DataDesc {
        std::type_info const &m_type_info_;
        size_t m_size_in_byte;
        std::shared_ptr<void> m_data_;
    };
    static constexpr int MAX_DOF = 10;
    int m_dof_ = 0;
    std::shared_ptr<int> m_idx_;
    std::map<std::string, DataDesc> m_data_;
    std::multimap<id_type, std::list<Bucket>> m_pool_;
    std::list<Bucket> m_free_bucket_;
};
ParticleBase::ParticleBase() : m_pimpl_(new pimpl_s){};

ParticleBase::~ParticleBase() = default;

void ParticleBase::DoInitialize(){};

ParticleBase::Bucket *ParticleBase::GetBucket(id_type s) { return nullptr; };
ParticleBase::Bucket const *ParticleBase::GetBucket(id_type s) const { return nullptr; };

void ParticleBase::Sort() {}
void ParticleBase::DeepSort(){};
}