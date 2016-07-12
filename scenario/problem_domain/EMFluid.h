/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include "../../src/field/Field.h"
#include "../../src/physics/PhysicalConstants.h"
#include "../../src/simulation/ProblemDomain.h"
#include "../../src/mesh/Mesh.h"
#include "../../src/mesh/MeshEntityRange.h"
#include "../../src/mesh/MeshModel.h"
#include "../../src/manifold/Calculus.h"

namespace simpla
{
using namespace mesh;


template<typename TM>
class EMFluid : public simulation::ProblemDomain
{
    typedef EMFluid<TM> this_type;
    typedef simulation::ProblemDomain base_type;

public:
    virtual bool is_a(std::type_info const &info) const
    {
        return typeid(this_type) == info || simulation::ProblemDomain::is_a(info);
    }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return "EMFluid<" + traits::type_id<TM>::name() + ">"; }

    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;
    mesh_type const *m;

    EMFluid(const mesh_type *mp) : base_type(mp), m(mp) { }

    virtual ~EMFluid() { }


    virtual void deploy();

    virtual void next_step(Real dt);

    virtual void sync(mesh::TransitionMap const &, simulation::ProblemDomain const &other);


    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    MeshEntityRange limiter_boundary;
    MeshEntityRange vertex_boundary;
    MeshEntityRange edge_boundary;
    MeshEntityRange face_boundary;

    MeshEntityRange plasma_region_volume;
    MeshEntityRange plasma_region_vertex;


    template<typename ValueType, size_t IFORM> using field_t =  Field<ValueType, TM, std::integral_constant<size_t, IFORM> >;;

    MeshEntityRange J_src_range;
    std::function<Vec3(Real, point_type const &, vector_type const &v)> J_src_fun;


    typedef field_t<scalar_type, FACE> TB;
    typedef field_t<scalar_type, EDGE> TE;
    typedef field_t<scalar_type, EDGE> TJ;
    typedef field_t<scalar_type, VERTEX> TRho;
    typedef field_t<vector_type, VERTEX> TJv;

    field_t<scalar_type, VERTEX> rho0{m};
    field_t<scalar_type, EDGE> E0/*   */{m};
    field_t<scalar_type, FACE> B0/*   */{m};
    field_t<vector_type, VERTEX> B0v/**/{m};
    field_t<scalar_type, VERTEX> BB/* */{m};
    field_t<vector_type, VERTEX> Ev/* */{m};
    field_t<vector_type, VERTEX> Bv/* */{m};
//
    field_t<scalar_type, FACE> B/*   */{m};
    field_t<scalar_type, EDGE> E/*   */{m};
    field_t<scalar_type, EDGE> J1/*  */{m};


    struct fluid_s
    {
        Real mass;
        Real charge;
        field_t<scalar_type, VERTEX> rho;
        field_t<vector_type, VERTEX> J;
    };

    std::map<std::string, fluid_s> m_fluid_sp_;

    fluid_s *
    add_particle(std::string const &name, Real mass, Real charge)
    {
        auto ins_res = m_fluid_sp_.emplace(std::make_pair(name, fluid_s{mass, charge, TRho{m}, TJv{m}}));
        fluid_s *res = nullptr;
        if (std::get<1>(ins_res)) { res = &(std::get<0>(ins_res)->second); }
        return res;
    }

};


template<typename TM>
void EMFluid<TM>::deploy()
{
    rho0.clear();
    E0.clear();
    B0.clear();
    B0v.clear();
    BB.clear();
    Ev.clear();
    Bv.clear();

    J1.clear();
    B.clear();
    E.clear();


    declare_global(&E, "E");
    declare_global(&B, "B");

    for (auto &sp:m_fluid_sp_)
    {
        declare_global(&(sp.second.rho), sp.first + "_rho");
        declare_global(&(sp.second.J), sp.first + "_J");
    }

}

template<typename TM> std::ostream &
EMFluid<TM>::print(std::ostream &os, int indent) const
{
    simulation::ProblemDomain::print(os, indent);
    os << std::setw(indent + 1) << " " << " ParticleAttribute= { " << std::endl;
    for (auto &sp:m_fluid_sp_)
    {
        os << std::setw(indent + 1) << " " << sp.first << " = { Mass=" << sp.second.mass << " , Charge = " <<
        sp.second.charge << "}," << std::endl;
    }
    os << std::setw(indent + 1) << " " << " }, " << std::endl;
    return os;

}


template<typename TM>
void EMFluid<TM>::sync(mesh::TransitionMap const &t_map, simulation::ProblemDomain const &other)
{
    auto const &E2 = *static_cast<field_t<scalar_type, mesh::EDGE> const *>( other.attribute("E"));
    auto const &B2 = *static_cast<field_t<scalar_type, mesh::FACE> const *>( other.attribute("B"));


    t_map.direct_map(mesh::EDGE,
                     [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2) { E[s1] = E2[s2]; });


    t_map.direct_map(mesh::FACE,
                     [&](mesh::MeshEntityId const &s1, mesh::MeshEntityId const &s2) { B[s1] = B2[s2]; });

}

template<typename TM>
void EMFluid<TM>::next_step(Real dt)
{

    VERBOSE << m->name() << " pushing!!" << std::endl;

    DEFINE_PHYSICAL_CONST

    if (J_src_fun)
    {
        Real current_time = m->time();

        auto f = J_src_fun;
        J_src_range.foreach(
                [&](mesh::id const &s)
                {
                    auto x0 = m->point(s);
                    auto v = J_src_fun(current_time, x0, J1(x0));
                    J1[s] += m->template sample<EDGE>(s, v);
                });
    }


    B -= curl(E) * (dt * 0.5);
    B.apply(face_boundary, [](mesh::MeshEntityId const &) -> Real { return 0.0; });

    E += (curl(B) * speed_of_light2 - J1 / epsilon0) * dt;
    E.apply(edge_boundary, [](mesh::MeshEntityId const &) -> Real { return 0.0; });

    field_t<vector_type, VERTEX> dE{m};

    if (m_fluid_sp_.size() > 0)
    {

        field_t<vector_type, VERTEX> Q{m};
        field_t<vector_type, VERTEX> K{m};

        field_t<scalar_type, VERTEX> a{m};
        field_t<scalar_type, VERTEX> b{m};
        field_t<scalar_type, VERTEX> c{m};

        a.clear();
        b.clear();
        c.clear();

        Q = map_to<VERTEX>(E) - Ev;


        for (auto &p :   m_fluid_sp_)
        {

            Real ms = p.second.mass;
            Real qs = p.second.charge;


            field_t<scalar_type, VERTEX> &ns = p.second.rho;

            field_t<vector_type, VERTEX> &Js = p.second.J;


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

        for (auto &p :   m_fluid_sp_)
        {
            Real ms = p.second.mass;
            Real qs = p.second.charge;
            field_t<scalar_type, VERTEX> &ns = p.second.rho;
            field_t<vector_type, VERTEX> &Js = p.second.J;


            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));

            K = dE * ns * qs * as;
            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
        }
        Ev += dE;

        E += map_to<EDGE>(Ev) - E;
    }

    B -= curl(E) * (dt * 0.5);
    B.apply(face_boundary, [](mesh::MeshEntityId const &) -> Real { return 0.0; });
}


}//namespace simpla  {
#endif //SIMPLA_EM_FLUID_H

