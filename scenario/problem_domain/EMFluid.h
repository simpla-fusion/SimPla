/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include <simpla/SIMPLA_config.h>
#include <simpla/physics/Field.h>
#include <simpla/physics/PhysicalConstants.h>
#include <simpla/mesh/Mesh.h>
#include <simpla/mesh/DomainBase.h>
#include <simpla/mesh/EntityRange.h>
#include <simpla/manifold/Calculus.h>
#include <simpla/model/Model.h>

namespace simpla
{
using namespace mesh;


template<typename TM>
class EMFluid : public DomainBase
{
    typedef EMFluid<TM> this_type;
    typedef DomainBase base_type;

public:
    virtual bool is_a(std::type_info const &info) const
    {
        return typeid(this_type) == info || DomainBase::is_a(info);
    }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return "EMFluid<" + traits::type_id<TM>::name() + ">"; }

    typedef TM mesh_type;

    typedef typename mesh_type::scalar_type scalar_type;

    using base_type::m;

    EMFluid() : base_type(new mesh_type) {}

    EMFluid(std::shared_ptr<mesh_type> mp) : base_type(mp) {}

    virtual ~EMFluid() {}

    virtual void deploy();

    virtual void next_step(Real dt);

    std::shared_ptr<mesh_type> mesh() const { return std::dynamic_pointer_cast<mesh_type>(base_type::mesh()); }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    EntityRange limiter_boundary;
    EntityRange vertex_boundary;
    EntityRange edge_boundary;
    EntityRange face_boundary;

    EntityRange plasma_region_volume;
    EntityRange plasma_region_vertex;


    template<typename ValueType, size_t IFORM> using field_t =  Field<ValueType, TM, std::integral_constant<size_t, IFORM> >;

    EntityRange J_src_range;
    std::function<Vec3(point_type const &, Real)> J_src_fun;

    EntityRange E_src_range;
    std::function<Vec3(point_type const &, Real)> E_src_fun;

    typedef field_t<scalar_type, FACE> TB;
    typedef field_t<scalar_type, EDGE> TE;
    typedef field_t<scalar_type, EDGE> TJ;
    typedef field_t<scalar_type, VERTEX> TRho;
    typedef field_t<vector_type, VERTEX> TJv;

    field_t<scalar_type, VERTEX> rho0{m};
    field_t<scalar_type, EDGE> E0{m};
    field_t<scalar_type, FACE> B0{m};
    field_t<vector_type, VERTEX> B0v/**/{m};
    field_t<scalar_type, VERTEX> BB/* */{m};
    field_t<vector_type, VERTEX> Ev/* */{m};
    field_t<vector_type, VERTEX> Bv/* */{m};
    field_t<vector_type, VERTEX> dE{m};

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
    mesh()->deploy();
    rho0.clear();
    E0.clear();
    B0.clear();
    BB.clear();
    Ev.clear();
    J1.clear();
    B.clear();
    E.clear();

    global_declare(&E, "E");
    global_declare(&B, "B");
    global_declare(&B0, "B0");
    global_declare(&B0v, "B0v");
    global_declare(&Ev, "Ev");
    global_declare(&dE, "dE");

    for (auto &sp:m_fluid_sp_)
    {
        global_declare(&(sp.second.rho), sp.first + "_rho");
        global_declare(&(sp.second.J), sp.first + "_J");
    }

}

template<typename TM>
std::ostream &
EMFluid<TM>::print(std::ostream &os, int indent) const
{
    DomainBase::print(os, indent);
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
void EMFluid<TM>::next_step(Real dt)
{

    VERBOSE << m->name() << " pushing!!" << std::endl;

    DEFINE_PHYSICAL_CONST

    if (J_src_fun) { J1.apply_function(_impl::plus_assign(), J_src_range, J_src_fun, m->time()); }

    if (E_src_fun) { E.apply_function(_impl::plus_assign(), E_src_range, E_src_fun, m->time()); }

    B -= curl(E) * (dt * 0.5);

    B.assign(face_boundary, 0);

    E += (curl(B) * speed_of_light2 - J1 / epsilon0) * dt;

    E.assign(edge_boundary, 0);


    if (m_fluid_sp_.size() > 0)
    {
        if (Ev.empty()) { Ev = map_to<VERTEX>(E); }
        if (B0v.empty())
        {
            B0v = map_to<VERTEX>(B0);
            BB = dot(B0v, B0v);
        }

        field_t<vector_type, VERTEX> Q{m};
        field_t<vector_type, VERTEX> K{m};

        field_t<scalar_type, VERTEX> a{m};
        field_t<scalar_type, VERTEX> b{m};
        field_t<scalar_type, VERTEX> c{m};

        a.clear();
        b.clear();
        c.clear();

        Q = map_to<VERTEX>(E) - Ev;

        K.clear();
        for (auto &p :m_fluid_sp_)
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

        for (auto &p : m_fluid_sp_)
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

    B.assign(face_boundary, 0);
}

}//namespace simpla  {
#endif //SIMPLA_EM_FLUID_H
