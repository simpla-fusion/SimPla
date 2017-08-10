//
// Created by salmon on 17-8-10.
//

#include "Particle.h"
#include "ParticleInitialLoad.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/engine/Mesh.h"

namespace simpla {

struct ParticlePool : public data::DataBlock {};

struct ParticleBase::pimpl_s {
    static constexpr int MAX_NUMBER_OF_PARTICLE_ATTRIBUTES = 10;
    engine::MeshBase const* m_mesh_;
    size_type m_size_ = 0;
    int m_dof_ = 3;
    ParticlePool* m_pool_ = nullptr;

    Real* m_data_[MAX_NUMBER_OF_PARTICLE_ATTRIBUTES];
};
ParticleBase::ParticleBase(engine::MeshBase const* m, int DOF) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_dof_ = DOF;
}
ParticleBase::~ParticleBase() {}
std::shared_ptr<data::DataTable> ParticleBase::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->Set("Properties", GetProperties());
    return res;
}
void ParticleBase::Deserialize(const std::shared_ptr<data::DataTable>& t) {
    if (t == nullptr) { return; }
    GetProperties()->Set(t->GetTable("Properties"));
}
void ParticleBase::PushData(data::DataBlock* dblk) {
    ASSERT(dblk->isA(typeid(ParticlePool)));
    m_pimpl_->m_pool_ = dynamic_cast<ParticlePool*>(dblk);
}

void ParticleBase::PopData(data::DataBlock* dblk) { m_pimpl_->m_pool_ = nullptr; }
int ParticleBase::GetNumberOfAttributes() const { return m_pimpl_->m_mesh_->GetNDIMS() + m_pimpl_->m_dof_; }
std::shared_ptr<ParticleBase::Bucket> ParticleBase::GetBucket(id_type s) { return nullptr; }
std::shared_ptr<ParticleBase::Bucket> ParticleBase::GetBucket(id_type s) const { return nullptr; }
size_type ParticleBase::Count(id_type s) const {
    size_type res = 0;
    if (s == NULL_ID) {
    } else {
        for (auto bucket = GetBucket(s); bucket != nullptr; bucket = bucket->next) { res += bucket->count; }
    }
    return res;
}

void ParticleBase::Sort() { UNIMPLEMENTED; }
void ParticleBase::DoInitialize() { UNIMPLEMENTED; }

void ParticleBase::InitialLoad(int const* rnd_dist_type, size_type rnd_offset) {
    int dist_type[GetNumberOfAttributes()];
    int ndims = m_pimpl_->m_mesh_->GetNDIMS();
    ASSERT(GetNumberOfAttributes() >= 2 * ndims);

    if (rnd_dist_type == nullptr) {
        for (int i = 0; i < m_pimpl_->m_mesh_->GetNDIMS(); ++i) { dist_type[i] = SP_RAND_UNIFORM; }
        for (int i = ndims; i < 2 * ndims; ++i) { dist_type[i] = SP_RAND_NORMAL; }

    } else {
        for (int i = 0; i < GetNumberOfAttributes(); ++i) { dist_type[i] = rnd_dist_type[i]; }
    }
    ParticleInitialLoad(m_pimpl_->m_data_, m_pimpl_->m_size_, 2 * ndims, dist_type, rnd_offset);
}

}  // namespace simpla {
