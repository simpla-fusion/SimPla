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
#include "../../src/mesh/MeshUtility.h"
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

//    template<typename _UOTHER_>
//    bool is_a() const { return is_a(typeid(_UOTHER_)); }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return "EMFluid<" + traits::type_id<TM>::name() + ">"; }


public:
    typedef TM mesh_type;
    typedef typename mesh_type::scalar_type scalar_type;
    mesh_type const *m;


    EMFluid(const mesh_type *mp) : base_type(mp), m(mp) { }

    virtual ~EMFluid() { }


    this_type &setup(ConfigParser const &options);

    virtual void deploy();

    virtual void next_step(Real dt);

    virtual void sync(mesh::TransitionMap const &, simulation::ProblemDomain const &other);

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

//    field_t<scalar_type, VERTEX> rho0{m};
//    field_t<scalar_type, EDGE> E0/*   */{m};
//    field_t<scalar_type, FACE> B0/*   */{m};
//    field_t<vector_type, VERTEX> B0v/**/{m};
//    field_t<scalar_type, VERTEX> BB/* */{m};
//    field_t<vector_type, VERTEX> Ev/* */{m};
//    field_t<vector_type, VERTEX> Bv/* */{m};
//
    field_t<scalar_type, FACE> B/*   */{m};
    field_t<scalar_type, EDGE> E/*   */{m};
    field_t<scalar_type, EDGE> J1/*  */{m};


    struct fluid_s
    {
        Real mass;
        Real charge;
        field_t<scalar_type, VERTEX> rho1;
        field_t<vector_type, VERTEX> J1;
    };

    std::map<std::string, fluid_s> fluid_sp;

    std::pair<typename std::map<std::string, fluid_s>::iterator, bool>
    add_particle(std::string const &name, Real mass, Real charge)
    {
        return fluid_sp.emplace(
                std::make_pair(name, fluid_s{mass, charge, TRho{*this, "n_" + name}, TJv{*this, "J_" + name}}));

    }

};

template<typename TM>
EMFluid<TM> &EMFluid<TM>::setup(ConfigParser const &options)
{
    if (options["Constraints"]["J"])
    {
        options["Constraints"]["J"]["Value"].as(&J_src_fun);
        mesh::select(*m, m->range(EDGE), options["Constraints"]["J"]["Box"].as<box_type>()).swap(J_src_range);
    }


    if (options["InitValue"])
    {
//        if (options["InitValue"]["B0"])
//        {
//            std::function<vector_type(point_type const &)> fun;
//            options["InitValue"]["B0"]["Value"].as(&fun);
//            m->range(FACE).foreach(
//                    [&](mesh::MeshEntityId const &s) { B0[s] = m->template sample<FACE>(s, fun(m->point(s))); });
//        }

        if (options["InitValue"]["B1"])
        {
            std::function<vector_type(point_type const &)> fun;
            options["InitValue"]["B1"]["Value"].as(&fun);
            m->range(FACE).foreach(
                    [&](mesh::MeshEntityId const &s) { B[s] = m->template sample<FACE>(s, fun(m->point(s))); });
        }

        if (options["InitValue"]["E1"])
        {
            std::function<vector_type(point_type const &)> fun;
            options["InitValue"]["E1"]["Value"].as(&fun);
            m->range(EDGE).foreach(
                    [&](mesh::MeshEntityId const &s) { E[s] = m->template sample<EDGE>(s, fun(m->point(s))); });
        }
    }

    if (options["Constraints"])
    {
        box_type b{{0, 0, 0},
                   {0, 0, 0}};

        options["Constraints"]["J"]["Box"].as(&b);

        J_src_range = m->range(b, mesh::EDGE);

        options["Constraints"]["J"]["Value"].as(&J_src_fun);
    }
    return *this;
}


template<typename TM>
void EMFluid<TM>::deploy()
{

    J1.clear();
    B.clear();
    E.clear();


    declare_global(&E, "E");
    declare_global(&B, "B");

//    VERBOSE << m->name() << " data =" << E.data().get() << std::endl;
//    if (m->name() == "Center")
//    {
//        E = 10;
//    }
//    else if (m->name() == "PML_0")
//    {
//        E = 20;
//    }
//    else if (m->name() == "PML_1")
//    {
//        E = 30;
//    }
//    else if (m->name() == "PML_2")
//    {
//        E = 40;
//    }
//    else if (m->name() == "PML_3")
//    {
//        E = 50;
//    }
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
//
//    if (m->name() == "Center")
//    {
//        E += 1.0;
//    }
//    else if (m->name() == "PML_0")
//    {
//        E += 2.0;
//    }
//    else if (m->name() == "PML_1")
//    {
//        E += 3.0;
//    }
//    else if (m->name() == "PML_2")
//    {
//        E += 4.0;
//    }
//    else if (m->name() == "PML_3")
//    {
//        E += 5.0;
//    }

    if (m->name() == "Center")
    {
        auto dims = m->dimensions();
        J1[MeshEntityId{0, 1, (dims[1] / 2 + 5) << 1, (dims[0] / 2 + 5) << 1}] +=
                std::sin(TWOPI * m->time() / (dt * 100.0));
    }

    B -= curl(E) * (dt);

    E += (curl(B) * speed_of_light2 - J1 / epsilon0) * dt;

//    if (J_src_fun)
//    {
//        Real current_time = m->time();
//
//        auto f = J_src_fun;
//        J_src_range.foreach(
//                [&](mesh::id const &s)
//                {
//                    auto x0 = m->point(s);
//                    auto v = J_src_fun(current_time, x0, J1(x0));
//                    J1[s] += m->template sample<EDGE>(s, v);
//                });
//    }
//
//
//    B1 -= curl(E1) * (dt*0.5);
//    B1.apply(face_boundary, [](mesh::MeshEntityId const &) -> Real { return 0.0; });
//    E1 += (curl(B1) * speed_of_light2 - J1 / epsilon0) * dt;
//    E1.apply(edge_boundary, [](mesh::MeshEntityId const &) -> Real { return 0.0; });
//
//    field_t<vector_type, VERTEX> dE{m};
//
//    if (fluid_sp.size() > 0)
//    {
//
//        field_t<vector_type, VERTEX> Q{m};
//        field_t<vector_type, VERTEX> K{m};
//
//        field_t<scalar_type, VERTEX> a{m};
//        field_t<scalar_type, VERTEX> b{m};
//        field_t<scalar_type, VERTEX> c{m};
//
//        a.clear();
//        b.clear();
//        c.clear();
//
//        Q = map_to<VERTEX>(E1) - Ev;
//
//
//        for (auto &p :   fluid_sp)
//        {
//
//            Real ms = p.second.mass;
//            Real qs = p.second.charge;
//
//
//            field_t<scalar_type, VERTEX> &ns = p.second.rho1;
//
//            field_t<vector_type, VERTEX> &Js = p.second.J1;;
//
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
//
//
//        }
//
//        a *= 0.5 * dt / epsilon0;
//        b *= 0.5 * dt / epsilon0;
//        c *= 0.5 * dt / epsilon0;
//        a += 1;
//
//
//        LOG_CMD(dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) /
//                     (b * b * BB + a * a));
//
//        for (auto &p :   fluid_sp)
//        {
//            Real ms = p.second.mass;
//            Real qs = p.second.charge;
//            field_t<scalar_type, VERTEX> &ns = p.second.rho1;
//            field_t<vector_type, VERTEX> &Js = p.second.J1;;
//
//
//            Real as = static_cast<Real>((dt * qs) / (2.0 * ms));
//
//            K = dE * ns * qs * as;
//            Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
//        }
//        Ev += dE;
//
//        E1 += map_to<EDGE>(Ev) - E1;
//    }
//
//    B1 -= curl(E1) * (dt * 0.5);
//    B1.apply(face_boundary, [](mesh::MeshEntityId const &) -> Real { return 0.0; });

}


}//namespace simpla  {
#endif //SIMPLA_EM_FLUID_H

