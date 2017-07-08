//
// Created by salmon on 17-6-8.
//

#ifndef SIMPLA_PARTICLE_H
#define SIMPLA_PARTICLE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/Attribute.h>
#include <simpla/algebra/EntityId.h>
#include <simpla/utilities/Range.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/engine/SPObject.h>
#include "ParticleBase.h"
#include "spParticle.h"
namespace simpla {

template <typename TM, int DOF = 6>
class Particle : public ParticleBase {
   private:
    typedef Particle<TM, DOF> field_type;
    SP_OBJECT_HEAD(field_type, ParticleBase);

   public:
    typedef TM mesh_type;
    static constexpr int iform = FIBER;

    static constexpr int NDIMS = mesh_type::NDIMS;

    typedef std::true_type prefer_pass_by_reference;
    typedef std::false_type is_expression;
    typedef std::false_type is_field;
    typedef std::true_type is_particle;

   private:
    mesh_type const* m_mesh_ = nullptr;
    EntityRange m_range_;

   public:
    template <typename... Args>
    explicit Particle(Args&&... args) : ParticleBase(DOF, std::forward<Args>(args)...){};

    Particle(this_type const& other) : ParticleBase(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {}

    Particle(this_type&& other) : ParticleBase(other), m_mesh_(other.m_mesh_), m_range_(other.m_range_) {}

    Particle(this_type const& other, EntityRange const& r) : ParticleBase(other), m_mesh_(other.m_mesh_), m_range_(r) {}

    ~Particle() override = default;

    this_type operator[](EntityRange const& d) const { return this_type(*this, d); }

    void swap(this_type& other) {
        m_range_.swap(other.m_range_);
        std::swap(m_mesh_, other.m_mesh_);
        ParticleBase::swap(other);
    }
    this_type& operator=(this_type const& other) {
        this_type(other).swap(*this);
        return *this;
    }

    //*****************************************************************************************************************

    void Update() override {
        if (m_mesh_ == nullptr) { m_mesh_ = dynamic_cast<mesh_type const*>(engine::Attribute::GetMesh()); }
        ASSERT(m_mesh_ != nullptr);
        ParticleBase::DoUpdate();
    }

    void TearDown() override {
        m_range_.reset();
        m_mesh_ = nullptr;
        ParticleBase::DoTearDown();
    }

};  // class Particle
}  // namespace simpla{

#endif  // SIMPLA_PARTICLE_H
