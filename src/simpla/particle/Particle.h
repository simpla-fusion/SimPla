//
// Created by salmon on 17-6-8.
//

#ifndef SIMPLA_PARTICLE_H
#define SIMPLA_PARTICLE_H

#include "ParticleBase.h"
#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/engine/Attribute.h"
namespace simpla {

template <typename TM, typename TV = Real, int DOF = 4>
class Particle : public engine::Attribute, public ParticleBase {
   private:
    typedef Particle<TM, Real, DOF> particle_type;
    SP_OBJECT_HEAD(particle_type, engine::Attribute);

   public:
    typedef TM mesh_type;
    static constexpr int iform = FIBER;

    static constexpr int NDIMS = mesh_type::NDIMS;

   private:
    mesh_type const* m_host_ = nullptr;
    EntityRange m_range_;
    std::shared_ptr<ParticleBase> m_data_;

   public:
    template <typename... Args>
    explicit Particle(mesh_type* grp, Args&&... args)
        : base_type(grp->GetMesh(), FIBER, std::integer_sequence<int, DOF>(), typeid(Real),
                    std::forward<Args>(args)...),
          m_host_(grp) {}

    ~Particle() override = default;

    explicit Particle(this_type const& other) : base_type(other), m_data_(other.m_data_), m_host_(other.m_host_) {}

    explicit Particle(this_type&& other) noexcept
        : base_type(std::forward<base_type>(other)), m_data_(other.m_data_), m_host_(other.m_host_) {}

    Particle& operator=(this_type&& other) = delete;

    this_type operator[](EntityRange const& d) const { return this_type(*this, d); }

    void swap(this_type& other) {
        m_range_.swap(other.m_range_);
        std::swap(m_host_, other.m_host_);
        ParticleBase::swap(other);
    }
    void DoInitialize() override {
        if (base_type::isNull()) {
            m_host_->GetMesh()->template initialize_data<IFORM>(&m_data_);
        } else {
            base_type::PushData(&m_data_);
        }

        traits::foreach (m_data_, [&](auto& a, auto&&... s) { a.Initialize(); });
    }

    void DoFinalize() override {
        base_type::PopData(&m_data_);
        traits::foreach (m_data_, [&](auto& a, auto&&... s) { a.Finalize(); });
    }

    void swap(this_type& other) {
        base_type::swap(other);
        m_data_.swap(other.m_data_);
        std::swap(m_host_, other.m_host_);
    }

    auto& Get() { return m_data_; }
    auto const& Get() const { return m_data_; }

};  // class Particle
}  // namespace simpla{

#endif  // SIMPLA_PARTICLE_H
