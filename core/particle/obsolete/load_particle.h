/**
 * @file load_particle.h
 *
 *  created on: 2013-12-21
 *      Author: salmon
 */

#ifndef LOAD_PARTICLE_H_
#define LOAD_PARTICLE_H_

#include <random>
#include <string>
#include <functional>

#include "field_comm.h"
#include "../field/loadField.h"

#include "../numeric/multi_normal_distribution.h"
#include "../numeric/rectangle_distribution.h"

#include "../physics/PhysicalConstants.h"


#include "../gtl/utilities/log.h"
#include "../gtl/utilities/utilities.h"
#include "../gtl/parallel/MPIAuxFunctions.h"
#include "particle_base.h"

namespace simpla {

template<typename TP, typename TDict, typename TModel, typename TN, typename TT>
std::shared_ptr<TP> load_particle(TDict const &dict, TModel const &model,
                                  TN const &ne0, TT const &T0)
{
    if (!dict
        || (TP::get_type_as_string()
            != dict["Type"].template as<std::string>()))
    {
        PARSER_ERROR(
                ("illegal particle configure!\"" + TP::get_type_as_string()
                 + " \"!= \" "
                 + dict["Type"].template as<std::string>("") + "\""))

        ;
    }

    typedef typename TP::mesh_type mesh_type;

    typedef typename mesh_type::coordinate_tuple coordinate_tuple;

    std::shared_ptr<TP> res(new TP(dict, model));

    std::function<Real(coordinate_tuple const &)> ns;

    std::function<Real(coordinate_tuple const &)> Ts;

    if (!T0.empty())
    {
        Ts = [&T0](coordinate_tuple x) -> Real
            { return T0(x); };
    }
    else if (dict["Temperature"].is_number())
    {
        Real T = dict["Temperature"].template as<Real>();
        Ts = [T](coordinate_tuple x) -> Real
            { return T; };
    }
    else if (dict["Temperature"].is_function())
    {
        Ts = dict["Temperature"].template as<
                std::function<Real(coordinate_tuple const &)>>();
    }

    if (!ne0.empty())
    {
        Real ratio = dict["Ratio"].template as<Real>(1.0);
        ns = [&ne0, ratio](coordinate_tuple x) -> Real
            { return ne0(x) * ratio; };
    }
    else if (dict["Density"].is_number())
    {
        Real n0 = dict["Density"].template as<Real>();
        ns = [n0](coordinate_tuple x) -> Real
            { return n0; };
    }
    else if (dict["Density"].is_function())
    {
        ns = dict["Density"].template as<
                std::function<Real(coordinate_tuple const &)>>();
    }

    size_t pic = dict["PIC"].template as<size_t>(100);

    auto range = model.select_by_config(TP::IForm, dict["Select"]);

    init_particle(res.get(), range, pic, ns, Ts);

    load_particle_constriant(res.get(), range, model, dict["Constraints"]);

    LOGGER << "Create Particles:[ Engine=" << res->get_type_as_string()
    << ", Number of Particles=" << res->size() << "]" << DONE;

    return std::dynamic_pointer_cast<ParticleBase>(res);

}

template<typename TP, typename TRange, typename TModel, typename TDict>
void load_particle_constriant(TP *p, TRange const &range, TModel const &model,
                              TDict const &dict)
{
    if (!dict)
        return;

    for (auto const &key_item : dict)
    {
        auto const &item = std::get<1>(key_item);

        auto r = model.select_by_config(range, item["Select"]);

        auto type = item["Type"].template as<std::string>("Modify");

        if (type == "Modify")
        {
            p->add_constraint([=]()
                                  { p->modify(r, item["Operations"]); });
        }
        else if (type == "Remove")
        {
            if (item["Operation"])
            {
                p->add_constraint([=]()
                                      { p->remove(r); });
            }
            else if (item["Condition"])
            {
                p->add_constraint([=]()
                                      { p->remove(r, item["Condition"]); });
            }
        }

    }
}

template<typename TR, typename TN, typename TT, typename TP>
void init_particle(TR const &domain, size_t pic, TN const &ns, TT const &Ts,
                   TP *p)
{
    typedef typename TP::engine_type engine_type;

    typedef TR domain_type;

    typedef typename domain_type::coordinate_tuple coordinate_tuple;

    static constexpr size_t ndims = domain_type::ndims;

    DEFINE_PHYSICAL_CONST

    std::mt19937 rnd_gen(ndims * 2);

    size_t number = domain.max_hash();

    std::tie(number, std::ignore) = sync_global_location(
            number * pic * ndims * 2);

    rnd_gen.discard(number);

    nTuple<Real, 3> x, v;

    Real inv_sample_density = 1.0 / pic;

    rectangle_distribution<3> x_dist;

    multi_normal_distribution<3> v_dist;

    auto mass = p->charge;

    for (auto s : domain)
    {

        for (int i = 0; i < pic; ++i)
        {
            x_dist(rnd_gen, &x[0]);

            v_dist(rnd_gen, &v[0]);

            x = domain.manifold_.coordinates_local_to_global(s, x);

            v *= std::sqrt(boltzmann_constant * Ts(x) / mass);

            p->emplace_back(x, v, ns(x) * inv_sample_density);
        }
//
//		auto & d = p->get(s);
//		d.splice(d.begin(), buffer);
    }

//	p->add(&buffer);
//	update_ghosts(p);
//	p->updateFields();

}
}  // namespace simpla

#endif /* LOAD_PARTICLE_H_ */
