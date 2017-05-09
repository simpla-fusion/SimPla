/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include "simpla/SIMPLA_config.h"
#include "simpla/algebra/all.h"
#include "simpla/engine/all.h"
#include "simpla/physics/PhysicalConstants.h"

namespace simpla {
using namespace algebra;
using namespace data;
using namespace engine;

template <typename TM>
class EMFluid : public engine::Domain {
    SP_OBJECT_HEAD(EMFluid<TM>, engine::Domain)
    typedef TM mesh_type;

   public:
    DOMAIN_HEAD(EMFluid, engine::Domain, mesh_type)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(shared_ptr<data::DataTable> const& cfg) override;

    void InitialCondition(Real time_now) override;
    void BoundaryCondition(Real time_now, Real dt) override;
    void Advance(Real time_now, Real dt) override;

    typedef field_type<FACE> TB;
    typedef field_type<EDGE> TE;
    typedef field_type<EDGE> TJ;
    typedef field_type<VERTEX> TRho;
    typedef field_type<VERTEX, 3> TJv;

    field_type<VERTEX> ne{this, "name"_ = "ne"};

    field_type<EDGE> E0{this};
    field_type<FACE> B0{this};
    field_type<VERTEX, 3> B0v{this, "name"_ = "B0"};
    field_type<VERTEX> BB{this, "name"_ = "BB"};
    field_type<VERTEX, 3> Ev{this, "name"_ = "Ev"};
    field_type<VERTEX, 3> Bv{this, "name"_ = "Bv"};
    field_type<VERTEX, 3> dE{this};

    field_type<FACE> B{this, "name"_ = "B"};
    field_type<EDGE> E{this, "name"_ = "E"};
    field_type<EDGE> J{this, "name"_ = "J"};

    struct fluid_s {
        Real mass;
        Real charge;
        std::shared_ptr<TRho> rho;
        std::shared_ptr<TJv> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;
    std::shared_ptr<fluid_s> AddSpecies(std::string const& name, std::shared_ptr<data::DataTable> const& d);
    std::map<std::string, std::shared_ptr<fluid_s>>& GetSpecies() { return m_fluid_sp_; };
};

template <typename TM>
bool EMFluid<TM>::is_registered = engine::Domain::RegisterCreator<EMFluid<TM>>();

template <typename TM>
std::shared_ptr<data::DataTable> EMFluid<TM>::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue<std::string>("Type", "EMFluid<" + TM::RegisterName() + ">");
    return res;
};
template <typename TM>
void EMFluid<TM>::Deserialize(shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr || cfg->GetTable("Species") == nullptr) { return; }
    auto sp = cfg->GetTable("Species");

    sp->Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> v) {
        if (!v->isTable()) { return; }
        auto t = std::dynamic_pointer_cast<data::DataTable>(v);
        AddSpecies(k, t);
    });
}

template <typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s> EMFluid<TM>::AddSpecies(std::string const& name,
                                                                     std::shared_ptr<data::DataTable> const& d) {
    auto sp = std::make_shared<fluid_s>();

    sp->mass = d->GetValue<double>("mass", d->GetValue<double>("m", 1) * SI_proton_mass);
    sp->charge = d->GetValue<double>("charge", d->GetValue<double>("Z", 1) * SI_elementary_charge);

    VERBOSE << "Add particle : {\"" << name << "\", mass = " << sp->mass / SI_proton_mass
            << " [m_p], charge = " << sp->charge / SI_elementary_charge << " [q_e] }" << std::endl;

    sp->rho = std::make_shared<TRho>(this, name + "_rho");
    sp->J = std::make_shared<TJv>(this, name + "_J");
    m_fluid_sp_.emplace(name, sp);
    return sp;
}

template <typename TM>
void EMFluid<TM>::InitialCondition(Real time_now) {
    Domain::InitialCondition(time_now);

    E.Clear();
    B.Clear();
    Ev.Clear();
    Bv.Clear();
    B0v.Clear();

    //    BB = inner_product(B0v, B0v);
}
template <typename TM>
void EMFluid<TM>::BoundaryCondition(Real time_now, Real dt) {
    //    auto brd = this->Boundary();
    //    if (brd == nullptr) { return; }
    //    E(brd) = 0;
    //    B(brd) = 0;
    //    auto antenna = this->SubMesh(m_antenna_);
    //    if (antenna != nullptr) {
    //        nTuple<Real, 3> k{1, 1, 1};
    //        Real omega = 1.0;
    //        E(antenna)
    //            .Assign([&](point_type const& x) {
    //                auto amp = std::sin(omega * time_now) * std::cos(dot(k, x));
    //                return nTuple<Real, 3>{amp, 0, 0};
    //            });
    //    }
}
template <typename TM>
void EMFluid<TM>::Advance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST
    //    B = B - curl(E) * (dt * 0.5);
    //    B[GetBoundaryRange(FACE)] = 0;
    //    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * dt;
    //    E[GetBoundaryRange(EDGE)] = 0;
    //    if (m_fluid_sp_.size() > 0) {
    //        field_type<VERTEX, 3> Q{this};
    //        field_type<VERTEX, 3> K{this};
    //
    //        field_type<VERTEX> a{this};
    //        field_type<VERTEX> b{this};
    //        field_type<VERTEX> c{this};
    //
    //        a.Clear();
    //        b.Clear();
    //        c.Clear();
    //
    //        Q = map_to<VERTEX>(E) - Ev;
    //        dE.Clear();
    //        K.Clear();
    //        for (auto& p : m_fluid_sp_) {
    //            Real ms = p.second->mass;
    //            Real qs = p.second->charge;
    //            auto& ns = *p.second->rho;
    //            auto& Js = *p.second->J;
    //
    //            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));
    //
    //            Q -= 0.5 * dt / epsilon0 * Js;
    //
    //            K = (Ev * qs * ns * 2.0 + cross(Js, B0v)) * as + Js;
    //
    //            Js = (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
    //
    //            Q -= 0.5 * dt / epsilon0 * Js;
    //
    //            a += qs * ns * (as / (BB * as * as + 1));
    //            b += qs * ns * (as * as / (BB * as * as + 1));
    //            c += qs * ns * (as * as * as / (BB * as * as + 1));
    //        }
    //
    //        a *= 0.5 * dt / epsilon0;
    //        b *= 0.5 * dt / epsilon0;
    //        c *= 0.5 * dt / epsilon0;
    //        a += 1;
    //
    //        dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a
    //        * a);
    //
    //        for (auto& p : m_fluid_sp_) {
    //            Real ms = p.second->mass;
    //            Real qs = p.second->charge;
    //            auto& ns = *p.second->rho;
    //            auto& Js = *p.second->J;
    //
    //            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));
    //
    //            K = dE * ns * qs * as;
    //            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
    //        }
    //        Ev += dE;
    //        E += map_to<EDGE>(Ev) - E;
    //    }
    //    B = B - curl(E) * (dt * 0.5);
    //    B[GetBoundaryRange(FACE)] = 0;
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
