/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 *
 * @example  em/em_plasma.cpp
 *    This is an example of EM plasma
 */

#include "plasma.h"

#include "../../core/gtl/utilities/utilities.h"
#include "../../core/parallel/parallel.h"
#include "../../core/io/io.h"

#include "../../core/manifold/pre_define/predefine.h"
#include "../../core/field/field.h"

#include "../../core/particle/particle.h"
#include "../../core/particle/particle_proxy.h"
#include "../../core/dataset/datatype_ext.h"
#include "../../core/particle/particle_generator.h"
#include "../../core/particle/pre_define/pic_boris.h"
#include "../../core/model/geqdsk.h"
#include "../../core/model/constraint.h"
#include "../../core/particle/particle_constraint.h"

namespace simpla
{


struct EMPlasma
{
    EMPlasma() { }

    ~EMPlasma() { }

    std::string description() { return ""; }

    virtual bool check_type(std::type_info const &info) { return info == typeid(EMPlasma); }

    void setup(int argc, char **argv);

    void next_time_step();

    void tear_down();

    void check_point();

    typedef Real scalar_type;

    typedef manifold::CylindricalManifold mesh_type;

    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::box_type box_type;
    typedef typename mesh_type::range_type range_type;
    typedef nTuple<scalar_type, 3> vector_type;

    mesh_type m;


    model::Surface<mesh_type> limiter_boundary;
    model::IdSet<mesh_type> vertex_boundary;
    model::IdSet<mesh_type> edge_boundary;
    model::IdSet<mesh_type> face_boundary;

    model::IdSet<mesh_type> J_src;

    std::function<Vec3(Real, point_type const &)> J_src_fun;

    traits::field_t<scalar_type, mesh_type, FACE> B0{m};
    traits::field_t<vector_type, mesh_type, VERTEX> B0v{m};
    traits::field_t<scalar_type, mesh_type, VERTEX> BB{m};

    traits::field_t<vector_type, mesh_type, VERTEX> Ev{m};
    traits::field_t<vector_type, mesh_type, VERTEX> Bv{m};

    traits::field_t<scalar_type, mesh_type, FACE> B1{m};
    traits::field_t<scalar_type, mesh_type, EDGE> E1{m};
    traits::field_t<scalar_type, mesh_type, EDGE> J1{m};

    traits::field_t<scalar_type, mesh_type, VERTEX> rho0{m};

    typedef traits::field_t<scalar_type, mesh_type, FACE> TB;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TE;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TJ;
    typedef traits::field_t<scalar_type, mesh_type, VERTEX> TRho;
    typedef traits::field_t<vector_type, mesh_type, VERTEX> TJv;


    typedef ParticleProxyBase<TE, TB, TJ, TRho> particle_proxy_type;

    typedef std::tuple<
            Real, //mass
            Real, //charge
            traits::field_t<scalar_type, mesh_type, VERTEX>, //rho_0
            traits::field_t<vector_type, mesh_type, VERTEX>,  //J1
            std::shared_ptr<particle_proxy_type>
    > particle_s;

    std::map<std::string, particle_s> particles;


    std::pair<typename std::map<std::string, particle_s>::iterator, bool>
    add_particle(std::string const &name, Real mass, Real charge, std::shared_ptr<particle_proxy_type> f = nullptr)
    {
        return particles.emplace(
                std::make_pair(
                        name,
                        std::make_tuple(
                                mass, charge,
                                traits::field_t<scalar_type, mesh_type, VERTEX>(m),
                                traits::field_t<vector_type, mesh_type, VERTEX>(m),
                                f)));

    }

//    struct particle_s
//    {
//        std::shared_ptr<ParticleProxyBase<TB, TE, TJ, TRho>> p;
//        Real mass;
//        Real charge;
//        TJ J1;
//        TRho rho1;
//        traits::field_t<scalar_type, mesh_type, VERTEX> rhos;
//        traits::field_t<vector_type, mesh_type, VERTEX> Js;
//    };


};

void EMPlasma::setup(int argc, char **argv)
{
    try
    {
        ConfigParser options;

        options.init(argc, argv);


        m.load(options);

        Real phi0 = std::get<0>(m.box())[2];

        Real phi1 = std::get<1>(m.box())[2];

        GEqdsk geqdsk;

        geqdsk.load(options["GEQDSK"].as<std::string>(""));

        auto box = geqdsk.limiter().box();


        std::get<0>(box)[2] = phi0;
        std::get<1>(box)[2] = phi1;

        m.box(box);

        m.deploy();

        m.open("em_plasma", "GEqdsk");


        MESSAGE << std::endl << "[ Configuration ]" << std::endl << m << std::endl;

        VERBOSE << "Clear fields" << std::endl;


        m.open_grid("back_ground");

        Ev.clear();

        B1.clear();
        E1.clear();
        J1.clear();

        B0.clear();

        serial::parallel_for(
                m.template range<FACE>(),
                [&](range_type const &r)
                {
                    for (auto const &s:r)
                    {
                        B0[s] = m.template sample<FACE>(s, geqdsk.B(m.point(s)));

                    }
                }
        );

        B0.sync();

        B0.save_as("B0");
        //        if (options["InitValue"]["B0"])
        //        {
        //          B0 = traits::make_field_function_from_config<scalar_type, FACE>(m, options["InitValue"]["B0"]);
        //        }

        rho0.clear();

        auto const &boundary = geqdsk.boundary();
        parallel::parallel_for(
                m.template range<VERTEX>(),
                [&](range_type const &r)
                {
                    for (auto const &s:r)
                    {
                        auto x = m.point(s);

                        if (boundary.within(x))
                        {
                            rho0[s] = m.template sample<VERTEX>(s, geqdsk.profile("ne", x));
                        }

                    }
                }
        );

        rho0.sync();

        rho0.save_as("rho0");


        B0v = map_to<VERTEX>(B0);

        BB = dot(B0, B0);


        {
            model::Cache<mesh_type> cache;

            model::update_cache(m, geqdsk.limiter(), &cache);

            model::get_cell_on_surface<EDGE>(m, cache, &edge_boundary);

            model::get_cell_on_surface<FACE>(m, cache, &face_boundary);

            model::get_cell_on_surface<VERTEX>(m, cache, &vertex_boundary);

        }

        {
            model::Cache<mesh_type> cache;

            model::update_cache(m, geqdsk.limiter(), &cache);

            model::get_cell_on_surface(m, cache, &limiter_boundary);

        }

        {

            auto dict = options["Constraints"]["J"];

            if (dict)
            {
                model::create_id_set(m,
                                     m.template make_range<EDGE>(
                                             m.index_box(dict["Box"].template as<box_type>())),
                                     &J_src);

                dict["Value"].as(&J_src_fun);
            }

        }

        {
            DEFINE_PHYSICAL_CONST;

            auto ps = options["Particles"];

            for (auto const &dict:ps)
            {
                auto res = add_particle(dict.first.template as<std::string>(),
                                        dict.second["mass"].template as<Real>(),
                                        dict.second["charge"].template as<Real>()
                );

                if (std::get<1>(res))
                {
                    std::string key = std::get<0>(res)->first;
                    auto &p = std::get<0>(res)->second;

                    if (dict.second["Density"])
                    {
                        std::get<2>(p) =
                                traits::make_field_function_from_config<scalar_type, VERTEX>(m, dict.second["Density"]);
                    }
                    else
                    {
                        std::get<2>(p) = rho0;
                    }


                    std::get<3>(p).clear();

                    std::get<2>(p).declare_as("n_" + key);
                    std::get<3>(p).declare_as("J_" + key);


                    if (dict.second["Type"].template as<std::string>() == "Bories")
                    {
                        std::get<4>(p) = particle_proxy_type::create<particle::BorisParticle<mesh_type>>(m);
                    }
                }


            }


        }




//        VERBOSE << "Generator Particles" << std::endl;
//
//        auto gen = particle::make_generator(ion, 1.0);
//
//        ion.generator(gen, options["PIC"].as<size_t>(10), 1.0);

        Ev = map_to<VERTEX>(E1);


    }
    catch (std::exception const &error)
    {
        THROW_EXCEPTION_RUNTIME_ERROR("Context setup error!", error.what());
    }

    Ev.declare_as("Ev");
    E1.declare_as("E1");
    B1.declare_as("B1");
    J1.declare_as("J1");

    m.close_grid();
    m.start_record("record");
}

void EMPlasma::tear_down()
{
    m.stop_record();
    m.open_grid("tear_down");

    B1.save_as("B1");

    m.close_grid();
    m.close();
}


void EMPlasma::check_point() { m.record(); }

void EMPlasma::next_time_step()
{
    VERBOSE << "Push one step" << std::endl;

    DEFINE_PHYSICAL_CONST

    Real dt = m.dt();

    Real t = m.time();


    LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

    B1.accept(face_boundary.range(), [&](id_type const &, Real &v) { v = 0; });

    J1.accept(J_src.range(),
              [&](id_type const &s, Real &v) { v += m.template sample<EDGE>(s, J_src_fun(t, m.point(s))); });

    LOG_CMD(E1 += (curl(B1) * speed_of_light2 - J1 / epsilon0) * dt);

    E1.accept(edge_boundary.range(), [&](id_type const &, Real &v) { v = 0; });

    //particle::absorb(ion, limiter_boundary);

    if (particles.size() > 0)
    {


        traits::field_t<vector_type, mesh_type, VERTEX> Q{m};
        traits::field_t<vector_type, mesh_type, VERTEX> K{m};


        traits::field_t<scalar_type, mesh_type, VERTEX> a{m};
        traits::field_t<scalar_type, mesh_type, VERTEX> b{m};
        traits::field_t<scalar_type, mesh_type, VERTEX> c{m};

        a.clear();
        b.clear();
        c.clear();


        Bv = map_to<VERTEX>(B1);

        Q = map_to<VERTEX>(E1) - Ev;

        for (auto &p :   particles)
        {

//        p.second.p->integral(&p.second.J1);

            Real ms, qs;

            std::tie(ms, qs, std::ignore, std::ignore, std::ignore) = p.second;

            traits::field_t<scalar_type, mesh_type, VERTEX> &ns = std::get<2>(p.second);

            traits::field_t<vector_type, mesh_type, VERTEX> &Js = std::get<3>(p.second);;


            Real as = (dt * qs) / (2.0 * ms);

            K = (Ev * ns * 2.0 + cross(Js, B0v)) * as + Js;

            Q -= 0.5 * dt / epsilon0
                 * (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as) / (BB * as * as + 1)) + Js;

            a += ns / BB * (as / (as * as + 1));
            b += ns / BB * (as * as / (as * as + 1));
            c += ns / BB * (as * as * as / (as * as + 1));

        }

        a *= 0.5 * dt / epsilon0;
        b *= 0.5 * dt / epsilon0;
        c *= 0.5 * dt / epsilon0;
        a += 1;


        Ev += (Q * a - cross(Q, B0v) * b +
               B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

        LOG_CMD(E1 += map_to<EDGE>(Ev) - E1);
    }
    LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

    B1.accept(face_boundary.range(), [&](id_type const &, Real &v) { v = 0; });

    m.next_time_step();
}

}

// namespace simpla
int main(int argc, char **argv)
{
    using namespace simpla;

    ConfigParser options;

    try
    {
        logger::init(argc, argv);

        parallel::init(argc, argv);

        io::init(argc, argv);

        options.init(argc, argv);
    }
    catch (std::exception const &error)
    {
        THROW_EXCEPTION_RUNTIME_ERROR("Initial error", error.what());
    }

    INFORM << ShowCopyRight() << std::endl;

    if (options["V"] || options["version"])
    {
        MESSAGE << "SIMPla " << ShowVersion();
        TheEnd(0);
        return TERMINATE;
    }
    else if (options["h"] || options["help"])
    {

        MESSAGE << " Usage: " << argv[0] << "   <options> ..." << std::endl << std::endl;

        MESSAGE << " Options:" << std::endl

        << "\t -h,\t--help            \t, Print a usage message and exit.\n"

        << "\t -v,\t--version         \t, Print version information exit. \n"

        << std::endl;


        TheEnd(0);

    }


    simpla::EMPlasma ctx;


    int num_of_step = options["number_of_step"].as<int>(20);

    int check_point = options["check_point"].as<int>(1);

    ctx.setup(argc, argv);


    int count = 0;

    ctx.check_point();

    MESSAGE << "====================================================" << std::endl;
    INFORM << "\t >>> START <<< " << std::endl;

    while (count < num_of_step)
    {
        ctx.next_time_step();

        if (count % check_point == 0)
            ctx.check_point();

        ++count;
    }
    ctx.tear_down();

    INFORM << "\t >>> Done <<< " << std::endl;
    MESSAGE << "====================================================" << std::endl;


    io::close();
    parallel::close();
    logger::close();

    return 0;
}

