/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/engine/Domain.h"
#include "simpla/engine/Model.h"
#include "simpla/physics/PhysicalConstants.h"
namespace simpla {

using namespace data;

template <typename THost>
class EMFluid {
    DOMAIN_POLICY_HEAD(EMFluid);

    void Serialize(data::DataTable* res) const;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg);
    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);
    void Advance(Real time_now, Real dt);

    Field<host_type, Real, VOLUME> ne{m_host_, "name"_ = "ne"};
    Field<host_type, Real, VOLUME, 3> B0v{m_host_, "name"_ = "B0v"};
    Field<host_type, Real, EDGE> E0{m_host_, "name"_ = "E0"};
    Field<host_type, Real, FACE> B0{m_host_, "name"_ = "B0"};
    Field<host_type, Real, VOLUME> BB{m_host_, "name"_ = "BB"};
    Field<host_type, Real, VOLUME, 3> Jv{m_host_, "name"_ = "Jv"};
    Field<host_type, Real, VOLUME, 3> Ev{m_host_, "name"_ = "Ev"};
    Field<host_type, Real, VOLUME, 3> Bv{m_host_, "name"_ = "Bv"};
    Field<host_type, Real, VOLUME, 3> dE{m_host_, "name"_ = "dE"};
    Field<host_type, Real, FACE> B{m_host_, "name"_ = "B"};
    Field<host_type, Real, EDGE> E{m_host_, "name"_ = "E"};
    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};
    Field<host_type, Real, VOLUME, 3> dumpE{m_host_, "name"_ = "dumpE"};
    Field<host_type, Real, VOLUME, 3> dumpB{m_host_, "name"_ = "dumpB"};
    Field<host_type, Real, VOLUME, 3> dumpJ{m_host_, "name"_ = "dumpJ"};

    struct fluid_s {
        Real mass = 1;
        Real charge = 1;
        Real ratio = 1;
        std::shared_ptr<Field<host_type, Real, VOLUME>> n;
        std::shared_ptr<Field<host_type, Real, VOLUME, 3>> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;
    std::shared_ptr<fluid_s> AddSpecies(std::string const& name, std::shared_ptr<data::DataTable> const& d);
    std::map<std::string, std::shared_ptr<fluid_s>>& GetSpecies() { return m_fluid_sp_; };

    std::string m_boundary_geo_obj_prefix_ = "PEC";
};

template <typename TM>
void EMFluid<TM>::Serialize(data::DataTable* res) const {
    for (auto& item : m_fluid_sp_) {
        auto t = std::make_shared<data::DataTable>();
        t->SetValue<double>("mass", item.second->mass / SI_proton_mass);
        t->SetValue<double>("Z", item.second->charge / SI_elementary_charge);
        t->SetValue<double>("ratio", item.second->ratio);

        res->Set("Species/" + item.first, t);
    }
};
template <typename TM>
void EMFluid<TM>::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr || cfg->GetTable("Species") == nullptr) { return; }
    auto sp = cfg->GetTable("Species");
    sp->Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> v) {
        if (!v->isTable()) { return; }
        auto t = std::dynamic_pointer_cast<data::DataTable>(v);
        AddSpecies(k, t);
    });
    m_boundary_geo_obj_prefix_ = cfg->GetValue<std::string>("BoundaryCondition/GeometryObject", "PEC");
}

template <typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s> EMFluid<TM>::AddSpecies(std::string const& name,
                                                                     std::shared_ptr<data::DataTable> const& d) {
    auto sp = std::make_shared<fluid_s>();
    sp->mass = d->GetValue<double>("mass", d->GetValue<double>("mass", 1)) * SI_proton_mass;
    sp->charge = d->GetValue<double>("charge", d->GetValue<double>("Z", 1)) * SI_elementary_charge;
    sp->ratio = d->GetValue<double>("ratio", d->GetValue<double>("ratio", 1));

    sp->n = std::make_shared<Field<host_type, Real, VOLUME>>(m_host_, "name"_ = name + "_n");
    sp->J = std::make_shared<Field<host_type, Real, VOLUME, 3>>(m_host_, "name"_ = name + "_J");
    m_fluid_sp_.emplace(name, sp);
    VERBOSE << "Add particle : {\"" << name << "\", mass = " << sp->mass / SI_proton_mass
            << " [m_p], charge = " << sp->charge / SI_elementary_charge << " [q_e] }" << std::endl;
    return sp;
}

template <typename TM>
void EMFluid<TM>::InitialCondition(Real time_now) {
    dumpE.Initialize();
    dumpB.Initialize();
    dumpJ.Initialize();
    E.Clear();
    B.Clear();
    J.Clear();

    Ev.Clear();
    Bv.Clear();

    ne.Initialize();
    B0v.Initialize();

    m_host_->GetModel().LoadProfile("ne", &ne);
    m_host_->GetModel().LoadProfile("B0", &B0v);

    BB = dot(B0v, B0v);

    for (auto& item : m_fluid_sp_) {
        if (item.second == nullptr) { continue; }
        item.second->n->Clear();
        *item.second->n = ne * item.second->ratio;
        item.second->J->Clear();
    }
    Ev = map_to<VOLUME>(E);
}
template <typename TM>
void EMFluid<TM>::BoundaryCondition(Real time_now, Real dt) {
    m_host_->FillBoundary(B, 0);
    m_host_->FillBoundary(E, 0);
    m_host_->FillBoundary(J, 0);
    //    m_host_->FillBoundary(dumpE, 0);
    //    m_host_->FillBoundary(dumpB, 0);
    //    m_host_->FillBoundary(dumpJ, 0);
}
template <typename TM>
void EMFluid<TM>::Advance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

//    B = B - curl(E) * (dt * 0.5);
//    m_host_->FillBoundary(B, 0);
//
//    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
//    m_host_->FillBoundary(E, 0);

    if (m_fluid_sp_.size() > 0) {
        Ev = map_to<VOLUME>(E);

        Field<host_type, Real, VOLUME, 3> Q{m_host_};
        Field<host_type, Real, VOLUME, 3> K{m_host_};

        Field<host_type, Real, VOLUME> a{m_host_};
        Field<host_type, Real, VOLUME> b{m_host_};
        Field<host_type, Real, VOLUME> c{m_host_};

        a.Clear();
        b.Clear();
        c.Clear();

        Q.Clear();
        K.Clear();

        dE.Clear();

        for (auto& p : m_fluid_sp_) {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto& ns = *p.second->n;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            Q -= (0.5 * dt / epsilon0) * Js;

            K = Js + cross(Js, B0v) * as + Ev * ns * (qs * 2.0 * as);

            Js = (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);

            Q -= (0.5 * dt / epsilon0) * Js;

            a += qs * ns * (as / (BB * as * as + 1));
            b += qs * ns * (as * as / (BB * as * as + 1));
            c += qs * ns * (as * as * as / (BB * as * as + 1));
        }

        a *= 0.5 * dt / epsilon0;
        b *= 0.5 * dt / epsilon0;
        c *= 0.5 * dt / epsilon0;
        a += 1;

        dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

        for (auto& p : m_fluid_sp_) {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto& ns = *p.second->n;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            K = dE * ns * qs * as;

            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
        }

        E = E + map_to<EDGE>(dE);
    }

//    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
//    m_host_->FillBoundary(E, 0);
//
//    B = B - curl(E) * (dt * 0.5);
//    m_host_->FillBoundary(B, 0);

    dumpE.DeepCopy(E);
    dumpB.DeepCopy(B);
    dumpJ.DeepCopy(J);
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
