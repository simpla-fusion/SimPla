/**
 * @file em_fluid.h
 * @author salmon
 * @date 2016-01-13.
 */

#ifndef SIMPLA_EM_FLUID_H
#define SIMPLA_EM_FLUID_H

#include "../model/Constraint.h"
#include "../manifold/pre_define/PreDefine.h"

namespace simpla { namespace phy_solver
{

template<typename TM>
class EMFluid
{

public:
    typedef TM mesh_type;

    typedef typename mesh_type::scalar_type scalar_type;

    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::box_type box_type;
    typedef typename mesh_type::range_type range_type;
    typedef nTuple<scalar_type, 3> vector_type;

    mesh_type &m;

    EMFluid(TM &m_p) : m(m_p) { }

    ~EMFluid() { }

    virtual void initialize(int argc, char **argv);

    virtual void next_time_step();

    virtual void tear_down();

    virtual void check_point();


    model::Surface<mesh_type> limiter_boundary;
    model::IdSet<mesh_type> vertex_boundary;
    model::IdSet<mesh_type> edge_boundary;
    model::IdSet<mesh_type> face_boundary;

    model::IdSet<mesh_type> plasma_region_volume;
    model::IdSet<mesh_type> plasma_region_vertex;
    model::IdSet<mesh_type> J_src;

    std::function<Vec3(Real, point_type const &)> J_src_fun;


    typedef traits::field_t<scalar_type, mesh_type, FACE> TB;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TE;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TJ;
    typedef traits::field_t<scalar_type, mesh_type, VERTEX> TRho;
    typedef traits::field_t<vector_type, mesh_type, VERTEX> TJv;

    traits::field_t<scalar_type, mesh_type, EDGE> E0{m, "E0"};
    traits::field_t<scalar_type, mesh_type, FACE> B0{m, "B0"};
    traits::field_t<vector_type, mesh_type, VERTEX> B0v{m, "B0v"};
    traits::field_t<scalar_type, mesh_type, VERTEX> BB{m, "BB"};

    traits::field_t<vector_type, mesh_type, VERTEX> Ev{m, "Ev"};
    traits::field_t<vector_type, mesh_type, VERTEX> Bv{m, "Bv"};

    traits::field_t<scalar_type, mesh_type, FACE> B1{m, "B1"};
    traits::field_t<scalar_type, mesh_type, EDGE> E1{m, "E1"};
    traits::field_t<scalar_type, mesh_type, EDGE> J1{m, "J1"};

    traits::field_t<scalar_type, mesh_type, VERTEX> rho0{m};


    struct fluid_s
    {
        Real mass;
        Real charge;
        traits::field_t<scalar_type, mesh_type, VERTEX> rho1;
        traits::field_t<vector_type, mesh_type, VERTEX> J1;
    };

    std::map<std::string, fluid_s> fluid_sp;

    std::pair<typename std::map<std::string, fluid_s>::iterator, bool>
    add_particle(std::string const &name, Real mass, Real charge)
    {
        return fluid_sp.emplace(std::make_pair(name, fluid_s{mass, charge, TRho{m, "n_" + name}, TJv{m, "J_" + name}}));

    }
};


template<typename TM>
void EMFluid<TM>::initialize(int argc, char **argv)
{

    ConfigParser options;

    options.init(argc, argv);



    Ev = map_to<VERTEX>(E1);


    out_stream.close_grid();
    out_stream.open_grid("record", io::XDMFStream::COLLECTION_TEMPORAL);

}

template<typename TM>
void EMFluid<TM>::tear_down()
{
    out_stream.close_grid();

    out_stream.open_grid("dump", io::XDMFStream::UNIFORM);

    out_stream.reference_topology_geometry("Main");

    out_stream.time(m.time());

    for (auto const &item:m.attributes())
    {
        if (!item.second.lock()->properties()["DisableXDMFOutput"])
        {
            out_stream.write(item.first, *std::dynamic_pointer_cast<base::AttributeObject>(item.second.lock()));

        } else
        {
            out_stream.hdf5().write(item.first, item.second.lock()->dump(), io::SP_RECORD);
        }
    }
    out_stream.close_grid();

    out_stream.close();
}

template<typename TM>
void EMFluid<TM>::check_point()
{
    if (!disable_field)
    {

        out_stream.open_grid(type_cast<std::string>(m_count), io::XDMFStream::UNIFORM);

        out_stream.reference_topology_geometry("Main");

        out_stream.time(m.time());

        for (auto const &item:m.attributes())
        {
            auto attr = item.second.lock();
            if (attr->properties()["EnableCheckPoint"])
            {
                if (!attr->properties()["IsParticle"])
                {
                    out_stream.write(item.first, *std::dynamic_pointer_cast<base::AttributeObject>(attr));
                }

            }
        }

        out_stream.close_grid();
    }
    if (!disable_particle)
    {


        for (auto const &item:m.attributes())
        {
            auto attr = item.second.lock();

            if (attr->properties()["EnableCheckPoint"])
            {
                if (attr->properties()["IsTestingParticle"])
                {
                    out_stream.hdf5().write(item.first, attr->checkpoint(), io::SP_BUFFER);
                }
                else if (attr->properties()["IsParticle"])
                {
                    out_stream.hdf5().write(item.first, attr->dump(), io::SP_RECORD);
                }


            }
        }
    }
    m.next_time_step();

    ++m_count;

}

template<typename TM>
void EMFluid<TM>::next_time_step()
{

    DEFINE_PHYSICAL_CONST

    Real dt = m.dt();

    Real t = m.time();

    LOGGER << " Time = [" << t << "] Count = [" << m_count << "]" << std::endl;


    if (!disable_field)
    {

        J1.clear();

        J1.accept(J_src.range(), [&](id_type s, Real &v) { J1.add(s, J_src_fun(t, m.point(s))); });

    }


    if (!disable_particle)
    {
        for (auto &p:particle_sp)
        {
            p.second->push(dt, m.time(), E0, B0);

            if (!disable_field) { p.second->integral(&J1); }
        }

        for (auto &p:testing_particle_sp)
        {
            p.second->push(dt, t, E0, B0);

            p.second->rehash();
        }


    }
    if (!disable_field)
    {


        LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

        B1.accept(face_boundary.range(), [&](id_type, Real &v) { v = 0; });


        LOG_CMD(E1 += (curl(B1) * speed_of_light2 - J1 / epsilon0) * dt);

        E1.accept(edge_boundary.range(), [&](id_type, Real &v) { v = 0; });


        traits::field_t<vector_type, mesh_type, VERTEX> dE{m};



        //particle::absorb(ion, limiter_boundary);
        if (fluid_sp.size() > 0)
        {

            traits::field_t<vector_type, mesh_type, VERTEX> Q{m};
            traits::field_t<vector_type, mesh_type, VERTEX> K{m};

            traits::field_t<scalar_type, mesh_type, VERTEX> a{m};
            traits::field_t<scalar_type, mesh_type, VERTEX> b{m};
            traits::field_t<scalar_type, mesh_type, VERTEX> c{m};

            a.clear();
            b.clear();
            c.clear();

            Q = map_to<VERTEX>(E1) - Ev;


            for (auto &p :   fluid_sp)
            {

                Real ms = p.second.mass;
                Real qs = p.second.charge;


                traits::field_t<scalar_type, mesh_type, VERTEX> &ns = p.second.rho1;

                traits::field_t<vector_type, mesh_type, VERTEX> &Js = p.second.J1;;


                Real as = (dt * qs) / (2.0 * ms);

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


            LOG_CMD(dE = (Q * a - cross(Q, B0v) * b + B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) /
                         (b * b * BB + a * a));

            for (auto &p :   fluid_sp)
            {
                Real ms = p.second.mass;
                Real qs = p.second.charge;
                traits::field_t<scalar_type, mesh_type, VERTEX> &ns = p.second.rho1;
                traits::field_t<vector_type, mesh_type, VERTEX> &Js = p.second.J1;;


                Real as = (dt * qs) / (2.0 * ms);

                K = dE * ns * qs * as;
                Js += (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as)) / (BB * as * as + 1);
            }
            Ev += dE;

            LOG_CMD(E1 += map_to<EDGE>(Ev) - E1);
        }


        LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

        B1.accept(face_boundary.range(), [&](id_type const &, Real &v) { v = 0; });
    }
    m.next_time_step();
}


}}//namespace simpla { namespace phy_solver {
#endif //SIMPLA_EM_FLUID_H
