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

   public:
    DOMAIN_HEAD(EMFluid, engine::Domain, TM)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;

    void InitialCondition(Real time_now) override;
    void BoundaryCondition(Real time_now, Real dt) override;
    void Advance(Real time_now, Real dt) override;

    //    field_type<VERTEX> ne{this, "name"_ = "ne"};
    //
    //    field_type<EDGE> E0{this};
    //    field_type<FACE> B0{this};
    //    field_type<VOLUME, 3> B0v{this, "name"_ = "B0v"};
    //    field_type<VOLUME> BB{this, "name"_ = "BB"};
    //    field_type<VOLUME, 3> Ev{this, "name"_ = "Ev"};
    //    field_type<VOLUME, 3> Bv{this, "name"_ = "Bv"};
    //    field_type<VOLUME, 3> Jv{this, "name"_ = "Jv"};
    //
    //    field_type<VOLUME, 3> dE{this};
    //
    //    field_type<FACE> B{this, "name"_ = "B"};
    //    field_type<EDGE> E{this, "name"_ = "E"};
    //    field_type<EDGE> J{this, "name"_ = "J"};

    DOMAIN_DECLARE_FIELD(ne, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(E0, EDGE, 1);
    DOMAIN_DECLARE_FIELD(B0, FACE, 1);
    DOMAIN_DECLARE_FIELD(B0v, VOLUME, 3);
    DOMAIN_DECLARE_FIELD(BB, VOLUME, 1);
    DOMAIN_DECLARE_FIELD(Ev, VOLUME, 3);
    DOMAIN_DECLARE_FIELD(Bv, VOLUME, 3);
    DOMAIN_DECLARE_FIELD(Jv, VOLUME, 3);
    DOMAIN_DECLARE_FIELD(dE, VOLUME, 3);
    DOMAIN_DECLARE_FIELD(B, FACE, 1);
    DOMAIN_DECLARE_FIELD(E, EDGE, 1);
    DOMAIN_DECLARE_FIELD(J, EDGE, 1);

    struct fluid_s {
        Real mass = 1;
        Real charge = 1;
        Real ratio = 1;
        std::shared_ptr<field_type<VOLUME>> rho;
        std::shared_ptr<field_type<VOLUME, 3>> J;
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

    for (auto& item : m_fluid_sp_) {
        auto t = std::make_shared<data::DataTable>();
        t->SetValue<double>("mass", item.second->mass / SI_proton_mass);
        t->SetValue<double>("Z", item.second->charge / SI_elementary_charge);
        t->SetValue<double>("ratio", item.second->ratio);

        res->Set("Species/" + item.first, t);
    }
    return res;
};
template <typename TM>
void EMFluid<TM>::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    DoInitialize();
    if (cfg == nullptr || cfg->GetTable("Species") == nullptr) { return; }
    auto sp = cfg->GetTable("Species");
    sp->Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> v) {
        if (!v->isTable()) { return; }
        auto t = std::dynamic_pointer_cast<data::DataTable>(v);
        AddSpecies(k, t);
    });
    Click();
}

template <typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s> EMFluid<TM>::AddSpecies(std::string const& name,
                                                                     std::shared_ptr<data::DataTable> const& d) {
    Click();
    auto sp = std::make_shared<fluid_s>();
    sp->mass = d->GetValue<double>("mass", d->GetValue<double>("mass", 1)) * SI_proton_mass;
    sp->charge = d->GetValue<double>("charge", d->GetValue<double>("Z", 1)) * SI_elementary_charge;
    sp->ratio = d->GetValue<double>("ratio", d->GetValue<double>("ratio", 1));

    sp->rho = std::make_shared<field_type<VOLUME>>(this, "name"_ = name + "_rho");
    sp->J = std::make_shared<field_type<VOLUME, 3>>(this, "name"_ = name + "_J");
    m_fluid_sp_.emplace(name, sp);
    VERBOSE << "Add particle : {\"" << name << "\", mass = " << sp->mass / SI_proton_mass
            << " [m_p], charge = " << sp->charge / SI_elementary_charge << " [q_e] }" << std::endl;
    return sp;
}

template <typename TM>
void EMFluid<TM>::InitialCondition(Real time_now) {
    DoSetUp();
    Domain::InitialCondition(time_now);

    E.Clear();
    B.Clear();
    Ev.Clear();
    Bv.Clear();
    J.Clear();
    Jv.Clear();

    BB = dot_v(B0v, B0v);

    for (auto& item : m_fluid_sp_) {
        if (item.second == nullptr) { continue; }

        *item.second->rho = ne * item.second->ratio;
        item.second->J->Clear();
    }
}
template <typename TM>
void EMFluid<TM>::BoundaryCondition(Real time_now, Real dt) {
    DoSetUp();

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
    DoSetUp();

    DEFINE_PHYSICAL_CONST

    B = B - curl(E) * (dt * 0.5);
    B[GetBoundaryRange(FACE)] = 0;

    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * dt;
    E[GetBoundaryRange(EDGE)] = 0;

    if (m_fluid_sp_.size() > 0) {
        field_type<VOLUME, 3> Q{this};
        field_type<VOLUME, 3> K{this};

        field_type<VOLUME> a{this};
        field_type<VOLUME> b{this};
        field_type<VOLUME> c{this};

        a.Clear();
        b.Clear();
        c.Clear();
        Q.Clear();
        dE.Clear();
        K.Clear();
        dE.DeepCopy(E);
        Q = map_to<VOLUME>(E) - Ev;

        for (auto& p : m_fluid_sp_) {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto& ns = *p.second->rho;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            Q += -0.5 * dt / epsilon0 * Js;

            K = (Ev * qs * ns * 2.0 + cross_v(Js, B0v)) * as + Js;

            Js = (K + cross_v(K, B0v) * as + B0v * (dot_v(K, B0v) * as * as)) / (BB * as * as + 1);

            Q += -0.5 * dt / epsilon0 * Js;

            a += qs * ns * (as / (BB * as * as + 1));
            b += qs * ns * (as * as / (BB * as * as + 1));
            c += qs * ns * (as * as * as / (BB * as * as + 1));
        }

        a *= 0.5 * dt / epsilon0;
        b *= 0.5 * dt / epsilon0;
        c *= 0.5 * dt / epsilon0;
        a += 1;
        dE = (Q * a - cross_v(Q, B0v) * b + B0v * (dot_v(Q, B0v) * (b * b - c * a) / (a + c * BB))) /
             (b * b * BB + a * a);

        for (auto& p : m_fluid_sp_) {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto& ns = *p.second->rho;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            K = dE * ns * qs * as;
            Js += (K + cross_v(K, B0v) * as + B0v * (dot_v(K, B0v) * as * as)) / (BB * as * as + 1);
        }
        Ev += dE;
        E += map_to<EDGE>(Ev) - E;
    }
    B = B - curl(E) * (dt * 0.5);
    B[GetBoundaryRange(FACE)] = 0;
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
