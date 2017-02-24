/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/Calculus.h>
#include <simpla/algebra/all.h>
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/Object.h>
#include <simpla/engine/Worker.h>
#include <simpla/physics/PhysicalConstants.h>

namespace simpla {
using namespace algebra;
using namespace data;

template <typename TM>
class EMFluid : public engine::Worker {
   public:
    SP_OBJECT_HEAD(EMFluid<TM>, engine::Worker)
    typedef TM mesh_type;
    typedef algebra::traits::scalar_type_t<mesh_type> scalar_type;

    EMFluid() {}
    ~EMFluid() {}

    std::shared_ptr<simpla::model::Model> m_model_;
    mesh_type* m_mesh_;

    virtual engine::MeshView* mesh() { return m_mesh_; };
    virtual engine::MeshView const* mesh() const { return m_mesh_; };

    virtual std::ostream& Print(std::ostream& os, int indent = 1) const;

    virtual void NextTimeStep(Real data_time, Real dt);
    virtual void Initialize();
    virtual void PreProcess();
    virtual void PostProcess();
    virtual void Finalize();

    virtual void SetPhysicalBoundaryConditions(Real time = 0){};
    virtual void SetPhysicalBoundaryConditionE(Real time = 0){};
    virtual void SetPhysicalBoundaryConditionB(Real time = 0){};

    template <int IFORM, int DOF = 1>
    using field_type = engine::FieldAttribute<TM, scalar_type, IFORM, DOF>;

    typedef field_type<FACE> TB;
    typedef field_type<EDGE> TE;
    typedef field_type<EDGE> TJ;
    typedef field_type<VERTEX> TRho;
    typedef field_type<VERTEX, 3> TJv;

    field_type<VERTEX> rho0{this, "rho0"};

    field_type<EDGE> E0{this, "E0"};
    field_type<FACE> B0{this, "B0"};
    field_type<VERTEX, 3> B0v{this, "B0v"};
    field_type<VERTEX> BB{this, "BB"};
    field_type<VERTEX, 3> Ev{this, "Ev"};
    field_type<VERTEX, 3> Bv{this, "Bv"};
    field_type<VERTEX, 3> dE{this, "dE"};

    field_type<FACE> B{this, "B", engine::CHECK};
    field_type<EDGE> E{this, "E", engine::CHECK};
    field_type<EDGE> J1{this, "J1", engine::CHECK};

    struct fluid_s {
        Real mass;
        Real charge;
        std::shared_ptr<TRho> rho;
        std::shared_ptr<TJv> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;

    std::shared_ptr<fluid_s> add_particle(std::string const& name, data::DataTable const& d);

    std::map<std::string, std::shared_ptr<fluid_s>>& particles() { return m_fluid_sp_; };
};

template <typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s> EMFluid<TM>::add_particle(std::string const& name,
                                                                       data::DataTable const& d) {
    Real mass;
    Real charge;

    if (d.has("mass")) {
        mass = d.as<double>("mass");
    } else if (d.has("m")) {
        mass = d.as<double>("m") * SI_proton_mass;
    } else {
        mass = SI_proton_mass;
    }

    if (d.has("charge")) {
        charge = d.as<double>("charge");
    } else if (d.has("Z")) {
        charge = d.as<double>("Z") * SI_elementary_charge;
    } else {
        charge = SI_elementary_charge;
    }

    VERBOSE << "Add particle : {\"" << name << "\", mass = " << mass / SI_proton_mass
            << " [m_p], charge = " << charge / SI_elementary_charge << " [q_e] }" << std::endl;
    auto sp = std::make_shared<fluid_s>();
    sp->mass = mass;
    sp->charge = charge;
    sp->rho = std::make_shared<TRho>(this, name + "_rho");
    sp->J = std::make_shared<TJv>(this, name + "_J");
    m_fluid_sp_.emplace(name, sp);
    return sp;
}

template <typename TM>
std::ostream& EMFluid<TM>::Print(std::ostream& os, int indent) const {
    os << std::setw(indent + 1) << " "
       << "ParticleAttribute=  " << std::endl
       << std::setw(indent + 1) << " "
       << "{ " << std::endl;
    for (auto& sp : m_fluid_sp_) {
        os << std::setw(indent + 1) << " " << sp.first << " = { Mass=" << sp.second->mass
           << " , Charge = " << sp.second->charge << "}," << std::endl;
    }
    os << std::setw(indent + 1) << " "
       << " }, " << std::endl;
    return os;
}

template <typename TM>
void EMFluid<TM>::PreProcess() {
    base_type::Update();
    //    if (E.isUpdated()) E.Clear();
    //    if (!B.isUpdated()) B.Clear();
    //    if (!B0.isUpdated()) { B0.Clear(); }
}

template <typename TM>
void EMFluid<TM>::PostProcess() {}

template <typename TM>
void EMFluid<TM>::Initialize() {
    if (m_fluid_sp_.size() > 0) {
        Ev = map_to<VERTEX>(E);
        B0v = map_to<VERTEX>(B0);
        BB = dot(B0v, B0v);
    }
}

template <typename TM>
void EMFluid<TM>::Finalize() {
    // do sth here
    PostProcess();
}

template <typename TM>
void EMFluid<TM>::NextTimeStep(Real data_time, Real dt) {
    PreProcess();
    DEFINE_PHYSICAL_CONST
    B -= curl(E) * (dt * 0.5);
    SetPhysicalBoundaryConditionB(data_time);
    E += (curl(B) * speed_of_light2 - J1 / epsilon0) * dt;
    SetPhysicalBoundaryConditionE(data_time);
    if (m_fluid_sp_.size() > 0) {
        field_type<VERTEX, 3> Q{this};
        field_type<VERTEX, 3> K{this};

        field_type<VERTEX> a{this};
        field_type<VERTEX> b{this};
        field_type<VERTEX> c{this};

        a.Clear();
        b.Clear();
        c.Clear();

        Q = map_to<VERTEX>(E) - Ev;
        dE.Clear();
        K.Clear();
        for (auto& p : m_fluid_sp_) {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto& ns = *p.second->rho;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            Q -= 0.5 * dt / epsilon0 * Js;

            K = (Ev * qs * ns * 2.0 + cross(Js, B0v)) * as + Js;

            Js = (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);

            Q -= 0.5 * dt / epsilon0 * Js;

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
            auto& ns = *p.second->rho;
            auto& Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            K = dE * ns * qs * as;
            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
        }
        Ev += dE;
        E += map_to<EDGE>(Ev) - E;
    }
    B -= curl(E) * (dt * 0.5);
    SetPhysicalBoundaryConditionB(data_time);
}

}  // namespace simpla  {
#endif  // SIMPLA_EM_FLUID_H
