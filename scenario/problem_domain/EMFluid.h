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

    mesh::Chart<TM> m_chart;

    EMFluid() {}

    ~EMFluid() {}

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual mesh::ChartBase *chart() { return &m_chart; };

    virtual mesh::ChartBase const *chart() const { return &m_chart; };

    virtual model::Model *model() { return m_model_.get(); };

    virtual model::Model const *model() const { return m_model_.get(); };

//    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m) { m_chart.move_to(m); }

//    virtual void deploy() {};

    virtual void next_time_step(Real data_time, Real dt);

    virtual void initialize(Real data_time = 0);

    virtual void set_physical_boundary_conditions(Real time);

    EntityIdRange limiter_boundary;
    EntityIdRange vertex_boundary;
    EntityIdRange edge_boundary;
    EntityIdRange face_boundary;

    EntityIdRange plasma_region_volume;
    EntityIdRange plasma_region_vertex;


    template<mesh::MeshEntityType IFORM, size_type DOF = 1>
    using field_type=Field<scalar_type, TM, index_const<IFORM>, index_const<DOF>>;

    EntityIdRange J_src_range;
    std::function<Vec3(point_type const &, Real)> J_src_fun;

    EntityIdRange E_src_range;
    std::function<Vec3(point_type const &, Real)> E_src_fun;

    typedef field_type<FACE> TB;
    typedef field_type<EDGE> TE;
    typedef field_type<EDGE> TJ;
    typedef field_type<VERTEX> TRho;
    typedef field_type<VERTEX, 3> TJv;

    field_type<VERTEX> rho0{&m_chart, "rho0"};
    field_type<EDGE> E0{&m_chart, "E0"};
    field_type<FACE> B0{&m_chart, "B0"};
    field_type<VERTEX, 3> B0v{&m_chart, "B0v"};
    field_type<VERTEX> BB{&m_chart, "BB"};
    field_type<VERTEX, 3> Ev{&m_chart, "Ev"};
    field_type<VERTEX, 3> Bv{&m_chart, "Bv"};
    field_type<VERTEX, 3> dE{&m_chart, "dE"};

    field_type<FACE> B/*   */{&m_chart, "B"};
    field_type<EDGE> E/*   */{&m_chart, "E"};
    field_type<EDGE> J1/*  */{&m_chart, "J1"};

    struct fluid_s
    {
        Real mass;
        Real charge;
        std::shared_ptr<TRho> rho;
        std::shared_ptr<TJv> J;
    };

    std::map<std::string, std::shared_ptr<fluid_s>> m_fluid_sp_;

    std::shared_ptr<fluid_s>
    add_particle(std::string const &name, Real mass, Real charge)
    {
        auto sp = std::make_shared<fluid_s>();
        sp->mass = mass;
        sp->charge = charge;
        sp->rho = std::make_shared<TRho>(&m_chart, name + "_rho");
        sp->J = std::make_shared<TJv>(&m_chart, name + "_J");
        m_fluid_sp_.emplace(std::make_pair(name, sp));
        return sp;
    }


    std::shared_ptr<model::Model> m_model_;

    virtual void connect_model(std::shared_ptr<model::Model> const &m) { m_model_ = m; };

};

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
void EMFluid<TM>::initialize(Real data_time)
{
    base_type::initialize(data_time);


    if (m_fluid_sp_.size() > 0)
    {
        Ev = map_to<VERTEX>(E);
        B0v = map_to<VERTEX>(B0);
        BB = dot(B0v, B0v);
    }
    for (auto &sp:m_fluid_sp_)
    {
        sp.second->rho->clear();
        sp.second->J->clear();
        sp.second->rho->assign([&](point_type const &x) { return std::sin(x[1]); });
    }
}

template<typename TM>
void EMFluid<TM>::set_physical_boundary_conditions(Real time)
{
    if (J_src_fun) { J1.assign([&](point_type const &x) { return J_src_fun(x, time); }, J_src_range); }
    if (E_src_fun) { E.assign([&](point_type const &x) { return E_src_fun(x, time); }, E_src_range); }
};


template<typename TM>
void EMFluid<TM>::next_time_step(Real data_time, Real dt)
{
    DEFINE_PHYSICAL_CONST
    B -= curl(E) * (dt * 0.5);
    B.assign(0, face_boundary);
    E += (curl(B) * speed_of_light2 - J1 / epsilon0) * dt;
    E.assign(0, edge_boundary);
    if (m_fluid_sp_.size() > 0)
    {
        field_type<VERTEX, 3> Q{&m_chart};
        field_type<VERTEX, 3> K{&m_chart};

        field_type<VERTEX> a{&m_chart};
        field_type<VERTEX> b{&m_chart};
        field_type<VERTEX> c{&m_chart};

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
    B.assign(0, face_boundary);
}

}//namespace simpla  {
#endif //SIMPLA_EM_FLUID_H

