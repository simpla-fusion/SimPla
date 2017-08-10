//
// Created by salmon on 17-8-7.
//

#include "ParticlePool.h"
namespace simpla {

struct ParticlePool::pimpl_s {
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
ParticlePool::ParticlePool() : m_pimpl_(new pimpl_s){};

ParticlePool::~ParticlePool() = default;

void ParticlePool::AddDOF(std::string const &s_name, std::type_info const &t_info, size_t t_size, void *ptr){};
void ParticlePool::Initialize(){};

ParticlePool::Bucket *ParticlePool::GetBucket(id_type s) { return nullptr; };
ParticlePool::Bucket const *ParticlePool::GetBucket(id_type s) const { return nullptr; };

void ParticlePool::Sort() {}
void ParticlePool::DeepSort() {}
Real **ParticlePool::GetAttributes() { return nullptr; }
Real const **ParticlePool::GetAttributes() const { return nullptr; }
size_type ParticlePool::GetSize() const { return 0; }

}  // namespace simpla{
