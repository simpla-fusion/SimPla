/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include <simpla/SIMPLA_config.h>

#include <simpla/physics/PhysicalConstants.h>
#include <simpla/mesh/EntityIdRange.h>
#include <simpla/manifold/Calculus.h>
#include <simpla/manifold/Worker.h>
#include <simpla/manifold/Chart.h>
#include <simpla/manifold/Field.h>


namespace simpla
{
using namespace mesh;


template<typename TM>
class EMFluid : public Worker
{

public:

    SP_OBJECT_HEAD(EMFluid<TM>, Worker);
    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;

    explicit EMFluid(std::shared_ptr<TM> const &c = nullptr) :
            Worker(c != nullptr ? c : std::make_shared<TM>()) {}

    ~EMFluid() {}

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void next_time_step(Real data_time, Real dt);

    virtual void pre_process();

    virtual void post_process();

    virtual void initialize(Real data_time = 0);

    virtual void finalize(Real data_time = 0);

    virtual void set_physical_boundary_conditions(Real time = 0) {};

    virtual void set_physical_boundary_conditions_E(Real time = 0) {};

    virtual void set_physical_boundary_conditions_B(Real time = 0) {};

    template<mesh::MeshEntityType IFORM, size_type DOF = 1>
    using field_type=Field<scalar_type, TM, index_const<IFORM>, index_const<DOF>>;


    typedef field_type<FACE> TB;
    typedef field_type<EDGE> TE;
    typedef field_type<EDGE> TJ;
    typedef field_type<VERTEX> TRho;
    typedef field_type<VERTEX, 3> TJv;

    field_type<VERTEX> rho0{m_chart_, "rho0", "CHECK"};

    field_type<EDGE> E0{m_chart_, "E0"};
    field_type<FACE> B0{m_chart_, "B0", "CHECK"};
    field_type<VERTEX, 3> B0v{m_chart_, "B0v"};
    field_type<VERTEX> BB{m_chart_, "BB"};
    field_type<VERTEX, 3> Ev{m_chart_, "Ev"};
    field_type<VERTEX, 3> Bv{m_chart_, "Bv"};
    field_type<VERTEX, 3> dE{m_chart_, "dE"};

    field_type<FACE> B/*   */{m_chart_, "B", "CHECK"};
    field_type<EDGE> E/*   */{m_chart_, "E", "CHECK"};
    field_type<EDGE> J1/*  */{m_chart_, "J1", "CHECK"};

    struct fluid_s
    {
        Real mass;
        Real charge;
        std::shared_ptr<TRho> rho;
        std::shared_ptr<TJv> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;

    std::shared_ptr<fluid_s> add_particle(std::string const &name, data::DataEntityTable const &d);

    std::map<std::string, std::shared_ptr<fluid_s>> &particles() { return m_fluid_sp_; };


};

template<typename TM>
std::shared_ptr<struct EMFluid<TM>::fluid_s>
EMFluid<TM>::add_particle(std::string const &name, data::DataEntityTable const &d)
{
    Real mass;
    Real charge;

    if (d.has("mass")) { mass = d.at("mass").as<double>(); }
    else if (d.has("m")) { mass = d.at("m").as<double>() * SI_proton_mass; }
    else { mass = SI_proton_mass; }

    if (d.has("charge")) { charge = d.at("charge").as<double>(); }
    else if (d.has("Z")) { charge = d.at("Z").as<double>() * SI_elementary_charge; }
    else { charge = SI_elementary_charge; }

    VERBOSE << "Add particle : {\"" << name << "\", mass = " << mass / SI_proton_mass << " [m_p], charge = "
            << charge / SI_elementary_charge << " [q_e] }" << std::endl;
    auto sp = std::make_shared<fluid_s>();
    sp->mass = mass;
    sp->charge = charge;
    sp->rho = std::make_shared<TRho>(m_chart_, name + "_rho");
    sp->J = std::make_shared<TJv>(m_chart_, name + "_J");
    m_fluid_sp_.emplace(std::make_pair(name, sp));
    return sp;
}


template<typename TM>
std::ostream &
EMFluid<TM>::print(std::ostream &os, int indent) const
{
    Worker::print(os, indent);

    os << std::setw(indent + 1) << " " << "ParticleAttribute=  " << std::endl
       << std::setw(indent + 1) << " " << "{ " << std::endl;
    for (auto &sp:m_fluid_sp_)
    {
        os << std::setw(indent + 1) << " " << sp.first << " = { Mass=" << sp.second->mass << " , Charge = " <<
           sp.second->charge << "}," << std::endl;
    }
    os << std::setw(indent + 1) << " " << " }, " << std::endl;
    return os;

}

template<typename TM>
void EMFluid<TM>::pre_process()
{
    if (is_valid()) { return; } else { base_type::pre_process(); }
    if (!E.is_valid()) E.clear();
    if (!B.is_valid()) B.clear();
    if (!B0.is_valid()) { B0.clear(); }
}

template<typename TM>
void EMFluid<TM>::post_process() { if (!is_valid()) { return; } else { base_type::post_process(); }}


template<typename TM>
void EMFluid<TM>::initialize(Real data_time)
{
    pre_process();

    if (m_fluid_sp_.size() > 0)
    {
        Ev = map_to<VERTEX>(E);
        B0v = map_to<VERTEX>(B0);
        BB = dot(B0v, B0v);
    }
    base_type::initialize(data_time, 0);
}


template<typename TM>
void EMFluid<TM>::finalize(Real data_time)
{
    // do sth here
    post_process();
}

template<typename TM>
void EMFluid<TM>::next_time_step(Real data_time, Real dt)
{
    pre_process();
    DEFINE_PHYSICAL_CONST
    B -= curl(E) * (dt * 0.5);
    set_physical_boundary_conditions_B(data_time);
    E += (curl(B) * speed_of_light2 - J1 / epsilon0) * dt;
    set_physical_boundary_conditions_E(data_time);
    if (m_fluid_sp_.size() > 0)
    {
        field_type<VERTEX, 3> Q{m_chart_};
        field_type<VERTEX, 3> K{m_chart_};

        field_type<VERTEX> a{m_chart_};
        field_type<VERTEX> b{m_chart_};
        field_type<VERTEX> c{m_chart_};

        a.clear();
        b.clear();
        c.clear();

        Q = map_to<VERTEX>(E) - Ev;
        dE.clear();
        K.clear();
        for (auto &p :m_fluid_sp_)
        {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto &ns = *p.second->rho;
            auto &Js = *p.second->J;

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


        dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) /
             (b * b * BB + a * a);

        for (auto &p : m_fluid_sp_)
        {
            Real ms = p.second->mass;
            Real qs = p.second->charge;
            auto &ns = *p.second->rho;
            auto &Js = *p.second->J;

            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            K = dE * ns * qs * as;
            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
        }
        Ev += dE;
        E += map_to<EDGE>(Ev) - E;
    }
    B -= curl(E) * (dt * 0.5);
    set_physical_boundary_conditions_B(data_time);
}

}//namespace simpla  {
#endif //SIMPLA_EM_FLUID_H

