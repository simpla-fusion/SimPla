/**
 * @file em_plasma.cpp
 * @author salmon
 * @date 2015-11-20.
 */

#include "em_plasma.h"

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

    typedef nTuple<scalar_type, 3> vector_type;

    mesh_type m;


    traits::field_t<vector_type, mesh_type, VERTEX> Bv{m};
    traits::field_t<vector_type, mesh_type, VERTEX> B0v{m};
    traits::field_t<vector_type, mesh_type, VERTEX> Ev{m};

    traits::field_t<scalar_type, mesh_type, VERTEX> rho0{m};
    traits::field_t<scalar_type, mesh_type, VERTEX> rho1{m};
    traits::field_t<scalar_type, mesh_type, FACE> B0{m};
    traits::field_t<scalar_type, mesh_type, FACE> B1{m};
    traits::field_t<scalar_type, mesh_type, EDGE> E1{m};
    traits::field_t<scalar_type, mesh_type, EDGE> pdE{m};

    typedef traits::field_t<scalar_type, mesh_type, FACE> TB;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TE;
    typedef traits::field_t<scalar_type, mesh_type, EDGE> TJ;
    typedef traits::field_t<scalar_type, mesh_type, VERTEX> TRho;


    TJ J1{m};

    particle::BorisParticle<mesh_type> ion{m};

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
//    std::map<std::string, particle_s> particles;

    model::Surface<mesh_type> limiter_boundary;
    model::IdSet<mesh_type> vertex_boundary;
    model::IdSet<mesh_type> edge_boundary;
    model::IdSet<mesh_type> face_boundary;

    model::IdSet<mesh_type> J_src;

    std::function<Vec3(Real, point_type const &)> J_src_fun;

};

void EMPlasma::setup(int argc, char **argv)
{
    try
    {
        ConfigParser options;

        options.init(argc, argv);


        m.load(options);

        GEqdsk geqdsk;

        geqdsk.load(options["GEQDSK"].as<std::string>(""));

        auto box = geqdsk.box();

        std::get<0>(box)[2] = 0;
        std::get<1>(box)[2] = TWOPI;

        m.box(box);

        m.deploy();

        MESSAGE << std::endl << "[ Configuration ]" << std::endl << m << std::endl;

        VERBOSE << "Clear fields" << std::endl;

        Bv.clear();
        B0v.clear();
        Ev.clear();

        B0.clear();
        B1.clear();
        E1.clear();
        pdE.clear();

        J1.clear();

        B0v = map_to<VERTEX>(B0);
        rho0.clear();
        rho1.clear();


        {
            model::Cache<mesh_type> cache;

            model::update_cache(m, geqdsk.limiter(), &cache);

            for (auto const &item:cache.range())
            {
                rho1[item.first] = std::get<0>(item.second);
                Bv[item.first] = std::get<1>(item.second);
            }

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


                model::create_id_set(m, m.template make_range<EDGE>(
                                             m.index_box(dict["Box"].template as<box_type>())),
                                     &J_src);

                dict["Value"].as(&J_src_fun);
            }

        }


        rho0.accept(vertex_boundary.range(), [&](id_type const &s, Real &v) { v = 1000; });

        E1.accept(edge_boundary.range(), [&](id_type const &s, Real &v) { v = 1000; });

        B1.accept(face_boundary.range(), [&](id_type const &s, Real &v) { v = 1000; });


//        VERBOSE << "Generator Particles" << std::endl;
//
//        auto gen = particle::make_generator(ion, 1.0);
//
//        ion.generator(gen, options["PIC"].as<size_t>(10), 1.0);


    }
    catch (std::exception const &error)
    {
        THROW_EXCEPTION_RUNTIME_ERROR("Context setup error!", error.what());
    }


    io::cd("/dump/");

    LOGGER << SAVE(E1) << std::endl;
    LOGGER << SAVE(B1) << std::endl;
    LOGGER << SAVE(Bv) << std::endl;
    LOGGER << SAVE(B0v) << std::endl;

    m.set_io_prefix("./em_plasma", "GEqdsk");
    m.set_io_time(m.time());
    m.register_dataset("E1", E1);

    m.write();

}

void EMPlasma::tear_down()
{
    io::cd("/tear_down/");
    LOGGER << SAVE(ion) << std::endl;
}


void EMPlasma::check_point()
{
    io::cd("/record/");

//    LOGGER << SAVE_RECORD(Bv) << std::endl;
//    LOGGER << SAVE_RECORD(B0v) << std::endl;
//    LOGGER << SAVE_RECORD(Ev) << std::endl;
//    LOGGER << SAVE_RECORD(B0) << std::endl;
    LOGGER << SAVE_RECORD(J1) << std::endl;
    LOGGER << SAVE_RECORD(B1) << std::endl;
    LOGGER << SAVE_RECORD(E1) << std::endl;

}

void EMPlasma::next_time_step()
{
    VERBOSE << "Push one step" << std::endl;

    DEFINE_PHYSICAL_CONST

    Real dt = m.dt();

    Real t = m.time();

//    ion.push(dt, 0, E1, B1);
//
//    J1.clear();
//
//    ion.integral(&J1);

    J1.accept(J_src.range(), [&](id_type const &s, Real &v)
    {
        v += m.template sample<EDGE>(s, J_src_fun(t, m.point(s)));
    });

    LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

    B1.accept(face_boundary.range(), [&](id_type const &, Real &v) { v = 0; });

    LOG_CMD(E1 += (curl(B1) * speed_of_light2 - J1 / epsilon0) * (dt * 0.5));

    E1.accept(edge_boundary.range(), [&](id_type const &, Real &v) { v = 0; });

    LOG_CMD(B1 -= curl(E1) * (dt * 0.5));

    B1.accept(face_boundary.range(), [&](id_type const &, Real &v) { v = 0; });

    particle::absorb(ion, limiter_boundary);
//
//
//    Ev = map_to<VERTEX>(E1);
//    Bv = map_to<VERTEX>(B1);
//
//    traits::field_t<scalar_type, mesh_type, VERTEX> BB{m};
//    BB = dot(B0, B0);
//
//    traits::field_t<vector_type, mesh_type, VERTEX> Q{m};
//    traits::field_t<vector_type, mesh_type, VERTEX> K{m};
//
//    Q = map_to<VERTEX>(pdE);
//
//    traits::field_t<scalar_type, mesh_type, VERTEX> a{m};
//    traits::field_t<scalar_type, mesh_type, VERTEX> b{m};
//    traits::field_t<scalar_type, mesh_type, VERTEX> c{m};
//
//    a.clear();
//    b.clear();
//    c.clear();
//
//    for (auto &p :   particles)
//    {
//
//        p.second.p->integral(&p.second.J1);
//
//        auto &rhos = p.second.rhos;
//        auto &Js = p.second.Js;
//
//        Real ms = p.second.mass;
//        Real qs = p.second.charge;
//
//        Real as = (dt * qs) / (2.0 * ms);
//
//        K = (Ev * rhos * 2.0 + cross(Js, B0v)) * as + Js;
//
//        Q -= 0.5 * dt / epsilon0
//             * (K + cross(K, B0v) * as + B0v * (dot(K, B0v) * as * as) / (BB * as * as + 1)) + Js;
//
//        a += rhos / BB * (as / (as * as + 1));
//        b += rhos / BB * (as * as / (as * as + 1));
//        c += rhos / BB * (as * as * as / (as * as + 1));
//
//    }
//
//    a *= 0.5 * dt / epsilon0;
//    b *= 0.5 * dt / epsilon0;
//    c *= 0.5 * dt / epsilon0;
//    a += 1;
//
//    auto dEv = traits::make_field<vector_type, VERTEX>(m);
//
//    dEv = (Q * a - cross(Q, B0v) * b +
//           B0v * (dot(Q, B0v) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);
//
//    Ev += dEv;
//
//    pdE = map_to<EDGE>(dEv);


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

