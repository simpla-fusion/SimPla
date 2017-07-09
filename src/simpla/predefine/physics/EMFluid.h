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

template <typename TM>
class EMFluid : public TM {
    SP_OBJECT_HEAD(EMFluid<TM>, TM)

   public:
    DOMAIN_HEAD(EMFluid, TM)(<#initializer#>)

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;

    void DoInitialCondition(Real time_now) override;
    void DoBoundaryCondition(Real time_now, Real dt) override;
    void DoAdvance(Real time_now, Real dt) override;

    Field<this_type, Real, VOLUME> ne{this, "name"_ = "ne"};
    Field<this_type, Real, VOLUME, 3> B0v{this, "name"_ = "B0v"};
    Field<this_type, Real, EDGE> E0{this, "name"_ = "E0"};
    Field<this_type, Real, FACE> B0{this, "name"_ = "B0"};
    Field<this_type, Real, VOLUME> BB{this, "name"_ = "BB"};
    Field<this_type, Real, VOLUME, 3> Jv{this, "name"_ = "Jv"};
    Field<this_type, Real, VOLUME, 3> Ev{this, "name"_ = "Ev"};
    Field<this_type, Real, VOLUME, 3> Bv{this, "name"_ = "Bv"};
    Field<this_type, Real, VOLUME, 3> dE{this, "name"_ = "dE"};
    Field<this_type, Real, FACE> B{this, "name"_ = "B"};
    Field<this_type, Real, EDGE> E{this, "name"_ = "E"};
    Field<this_type, Real, EDGE> J{this, "name"_ = "J"};
    Field<this_type, Real, VOLUME, 3> dumpE{this, "name"_ = "dumpE"};
    Field<this_type, Real, VOLUME, 3> dumpB{this, "name"_ = "dumpB"};
    Field<this_type, Real, VOLUME, 3> dumpJ{this, "name"_ = "dumpJ"};

    struct fluid_s {
        Real mass = 1;
        Real charge = 1;
        Real ratio = 1;
        std::shared_ptr<Field<mesh_type, Real, VOLUME>> n;
        std::shared_ptr<Field<mesh_type, Real, VOLUME, 3>> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;
    std::shared_ptr<fluid_s> AddSpecies(std::string const& name, std::shared_ptr<data::DataTable> const& d);
    std::map<std::string, std::shared_ptr<fluid_s>>& GetSpecies() { return m_fluid_sp_; };

    std::string m_boundary_geo_obj_prefix_ = "PEC";
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
    m_boundary_geo_obj_prefix_ = cfg->GetValue<std::string>("DoBoundaryCondition/GeometryObject", "PEC");
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

    sp->n = std::make_shared<Field<mesh_type, Real, VOLUME>>(this, "name"_ = name + "_n");
    sp->J = std::make_shared<Field<mesh_type, Real, VOLUME, 3>>(this, "name"_ = name + "_J");
    m_fluid_sp_.emplace(name, sp);
    VERBOSE << "Add particle : {\"" << name << "\", mass = " << sp->mass / SI_proton_mass
            << " [m_p], charge = " << sp->charge / SI_elementary_charge << " [q_e] }" << std::endl;
    return sp;
}

template <typename TM>
void EMFluid<TM>::DoInitialCondition(Real time_now) {
    Domain::InitialCondition(time_now);

    dumpE.Clear();
    dumpB.Clear();
    dumpJ.Clear();
    E.Clear();
    B.Clear();
    J.Clear();

    Ev.Clear();
    Bv.Clear();

    B0v.Update();
    ne.Update();

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
void EMFluid<TM>::DoBoundaryCondition(Real time_now, Real dt) {
    FillBoundary(B, 0);
    FillBoundary(E, 0);
    FillBoundary(J, 0);
    FillBoundary(dumpE, 0);
    FillBoundary(dumpB, 0);
}
template <typename TM>
void EMFluid<TM>::DoAdvance(Real time_now, Real dt) {
    DEFINE_PHYSICAL_CONST

    B = B - curl(E) * (dt * 0.5);
    FillBoundary(B, 0);

    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
    FillBoundary(E, 0);

    if (m_fluid_sp_.size() > 0) {
        Ev = map_to<VOLUME>(E);

        Field<mesh_type, Real, VOLUME, 3> Q{this};
        Field<mesh_type, Real, VOLUME, 3> K{this};

        Field<mesh_type, Real, VOLUME> a{this};
        Field<mesh_type, Real, VOLUME> b{this};
        Field<mesh_type, Real, VOLUME> c{this};

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

            K = Js + cross_v(Js, B0v) * as + Ev * ns * (qs * 2.0 * as);

            Js = (K + cross_v(K, B0v) * as + B0v * (dot_v(K, B0v) * as * as)) / (BB * as * as + 1);

            Q -= (0.5 * dt / epsilon0) * Js;

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
            auto& ns = *p.second->n;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            K = dE * ns * qs * as;

            Js += (K + cross_v(K, B0v) * as + B0v * (dot_v(K, B0v) * as * as)) / (BB * as * as + 1);
        }

        E += map_to<EDGE>(dE);
    }

    E = E + (curl(B) * speed_of_light2 - J / epsilon0) * 0.5 * dt;
    FillBoundary(E, 0);

    B = B - curl(E) * (dt * 0.5);
    FillBoundary(B, 0);

    dumpE.DeepCopy(E);
    dumpB.DeepCopy(B);
    dumpJ.DeepCopy(J);
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
