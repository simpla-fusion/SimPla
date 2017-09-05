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
    SP_ENGINE_POLICY_HEAD(EMFluid);

    void InitialCondition(Real time_now);
    void BoundaryCondition(Real time_now, Real time_dt);
    void Advance(Real time_now, Real dt);

    Field<host_type, Real, CELL> ne{m_host_, "name"_ = "ne"};
    Field<host_type, Real, CELL, 3> B0v{m_host_, "name"_ = "B0v"};

    Field<host_type, Real, EDGE> E0{m_host_, "name"_ = "E0"};
    Field<host_type, Real, FACE> B0{m_host_, "name"_ = "B0"};
    Field<host_type, Real, CELL> BB{m_host_, "name"_ = "BB"};
    Field<host_type, Real, CELL, 3> Jv{m_host_, "name"_ = "Jv"};
    Field<host_type, Real, CELL, 3> Ev{m_host_, "name"_ = "Ev"};
    Field<host_type, Real, CELL, 3> Bv{m_host_, "name"_ = "Bv"};
    Field<host_type, Real, CELL, 3> dE{m_host_, "name"_ = "dE"};
    Field<host_type, Real, FACE> B{m_host_, "name"_ = "B"};
    Field<host_type, Real, EDGE> E{m_host_, "name"_ = "E"};
    Field<host_type, Real, EDGE> J{m_host_, "name"_ = "J"};
    Field<host_type, Real, CELL, 3> dumpE{m_host_, "name"_ = "dumpE"};
    Field<host_type, Real, CELL, 3> dumpB{m_host_, "name"_ = "dumpB"};
    Field<host_type, Real, CELL, 3> dumpJ{m_host_, "name"_ = "dumpJ"};

    //    void TagRefinementCells(Real time_now);

    struct fluid_s {
        Real mass = 1;
        Real charge = 1;
        Real ratio = 1;
        std::shared_ptr<Field<host_type, Real, CELL>> n;
        std::shared_ptr<Field<host_type, Real, CELL, 3>> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;
    std::shared_ptr<fluid_s> AddSpecies(std::string const& name, std::shared_ptr<data::DataNode> d);
    std::map<std::string, std::shared_ptr<fluid_s>>& GetSpecies() { return m_fluid_sp_; };
};

template <typename TM>
std::shared_ptr<data::DataNode> EMFluid<TM>::Serialize() const {
    auto res = data::DataNode::New();
    for (auto& item : m_fluid_sp_) {
        res->SetValue("Species/" + item.first + "/mass", item.second->mass / SI_proton_mass);
        res->SetValue("Species/" + item.first + "/Z", item.second->charge / SI_elementary_charge);
        res->SetValue("Species/" + item.first + "/ratio", item.second->ratio);
    }
    return res;
};
template <typename TM>
void EMFluid<TM>::Deserialize(std::shared_ptr<data::DataNode>const & cfg) {
    cfg->Get("Species")->Foreach([&](std::string const& k, std::shared_ptr<data::DataNode> v) {
        return AddSpecies(k, v) != nullptr ? 1 : 0;
    });
}

template <typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s> EMFluid<TM>::AddSpecies(std::string const& name,
                                                                     std::shared_ptr<data::DataNode> d) {
    if (d == nullptr) { return nullptr; }

    auto sp = std::make_shared<fluid_s>();
    sp->mass = d->GetValue<double>("mass", d->GetValue<double>("mass", 1)) * SI_proton_mass;
    sp->charge = d->GetValue<double>("charge", d->GetValue<double>("Z", 1)) * SI_elementary_charge;
    sp->ratio = d->GetValue<double>("ratio", d->GetValue<double>("ratio", 1));

    sp->n = std::make_shared<Field<host_type, Real, CELL>>(m_host_, "name"_ = name + "_n");
    sp->J = std::make_shared<Field<host_type, Real, CELL, 3>>(m_host_, "name"_ = name + "_J");
    m_fluid_sp_.emplace(name, sp);
    VERBOSE << "AddEntity particle : {\"" << name << "\", mass = " << sp->mass / SI_proton_mass
            << " [m_p], charge = " << sp->charge / SI_elementary_charge << " [q_e] }" << std::endl;
    return sp;
}
//
// template <typename TM>
// void EMFluid<TM>::TagRefinementCells(Real time_now) {
//    m_host_->GetMesh()->TagRefinementCells(m_host_->GetMesh()->GetRange(m_host_->GetName() + "_BOUNDARY_3"));
//}
template <typename TM>
void EMFluid<TM>::InitialCondition(Real time_now) {
    E.Clear();
    B.Clear();
    J.Clear();

    Ev.Clear();
    Bv.Clear();

    ne.Clear();

    if (m_host_->GetModel() != nullptr) { m_host_->GetModel()->LoadProfile("ne", &ne); }

    return;
    BB = dot(B0v, B0v);

    for (auto& item : m_fluid_sp_) {
        if (item.second == nullptr) { continue; }
        item.second->n->Clear();
        *item.second->n = ne * item.second->ratio;
        item.second->J->Clear();
    }
    Ev = map_to<CELL>(E);
}
template <typename TM>
void EMFluid<TM>::BoundaryCondition(Real time_now, Real dt) {
    m_host_->FillBoundary(B, 0);
    m_host_->FillBoundary(E, 0);
    m_host_->FillBoundary(J, 0);
}
template <typename TM>
void EMFluid<TM>::Advance(Real time_now, Real dt) {
    return;
    DEFINE_PHYSICAL_CONST

    //    B = B - curl(E) * (dt * 0.5);
    //    m_host_->FillBoundary(B, 0);
    //
    //    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
    //    m_host_->FillBoundary(E, 0);

    if (m_fluid_sp_.size() <= 0) { return; }
    Ev = map_to<CELL>(E);

    Field<host_type, Real, CELL, 3> Q{m_host_};
    Field<host_type, Real, CELL, 3> K{m_host_};

    Field<host_type, Real, CELL> a{m_host_};
    Field<host_type, Real, CELL> b{m_host_};
    Field<host_type, Real, CELL> c{m_host_};

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

        auto as = static_cast<Real>((dt * qs) / (2.0 * ms));

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

        auto as = static_cast<Real>((dt * qs) / (2.0 * ms));

        K = dE * ns * qs * as;

        Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
    }

    E = E + map_to<EDGE>(dE);

    //    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
    //    m_host_->FillBoundary(E, 0);
    //
    //    B = B - curl(E) * (dt * 0.5);
    //    m_host_->FillBoundary(B, 0);
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
