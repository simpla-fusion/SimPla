/**
 * @file surface.h
 * @author salmon
 * @date 2015-11-30.
 */

#ifndef SIMPLA_SURFACE_H
#define SIMPLA_SURFACE_H

#include "../parallel/parallel.h"
#include "../geometry/geo_object.h"
#include "../manifold/manifold_traits.h"
#include "../geometry/geo_algorithm.h"

namespace simpla { namespace model
{
namespace _impl
{
template<typename ...> struct constraint_type_choice;
template<typename TM>
struct constraint_type_choice<TM>
{
    typedef parallel::concurrent_unordered_set<typename TM::id_type> type;
};
template<typename TM, typename Arg>
struct constraint_type_choice<TM, Arg>
{
    typedef parallel::concurrent_hash_map<typename TM::id_type, Arg> type;
};

template<typename TM, typename Arg0, typename ...Args>
struct constraint_type_choice<TM, Arg0, Args ...>
{
    typedef parallel::concurrent_hash_map<typename TM::id_type, std::tuple<Arg0, Args...>> type;
};
}

template<typename TM, typename ...Args> using Constraint=typename _impl::constraint_type_choice<TM, Args...>::type;

template<typename TM> using IdSet=Constraint<TM>;

template<typename TM> using Surface=Constraint<TM, Real, typename TM::point_type, typename TM::vector_type>;

template<typename TM> using Cache = Surface<TM>;


template<typename TM, int IFORM = VERTEX>
void create_cache(TM const &m, geometry::Object const &geo, Cache<TM> *cache)
{

    typedef TM mesh_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename mesh_type::vector_type vector_type;

    typedef typename mesh_type::id_type id_type;


    parallel::parallel_for(m.template range<IFORM>(),
                           [&](typename TM::range_type const &r)
                           {
                               for (auto const &s:r)
                               {
                                   point_type x = m.point(s);
                                   Vec3 v;
                                   Real d = geo.normals(&x, &v);
                                   typename Cache<TM>::value_type tmp{s, std::make_tuple(d, x, v)};
                                   cache->insert(tmp);
                               }
                           });


}

template<typename TM, typename Func>
void on_surface(TM const &m, Cache<TM> const &cache, Func const &func)
{

    typedef TM mesh_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename mesh_type::vector_type vector_type;

    typedef typename mesh_type::id_type id_type;


    size_t MASK = m.id_mask();

    serial::parallel_for(
            cache.range(),
            [&](typename Cache<TM>::const_range_type const &r)
            {
                for (auto const &item: r)
                {
                    id_type v_s = item.first + mesh_type::_DA;

                    id_type p[mesh_type::MAX_NUM_OF_NEIGHBOURS];

                    int num = m.get_adjacent_cells(VERTEX, mesh_type::TAG_VOLUME, v_s, p);

                    int count = 0;

                    for (int i = 0; i < num; ++i)
                    {
                        typename Cache<TM>::const_accessor acc;

                        if (cache.find(acc, (p[i] & MASK)))
                        {
                            if (std::get<0>(acc->second) > 0) { ++count; }
                        }
                    }
                    if ((count > 0) && (count < num))
                    {
                        func(item);
                    }

                }

            }

    );


}

template<typename TM, typename ...Args>
void get_surface(TM const &m, geometry::Object const &geo, Args &&...args)
{

    Cache<TM> cache;

    create_cache(geo, m, &cache);

    get_surface(cache, cache, std::forward<Args>(args)...);
}

template<typename TM>
void get_surface(TM const &m, Cache<TM> const &cache, Surface<TM> *surface)
{
    on_surface(m, cache,
               [&](typename Cache<TM>::value_type const &item) { surface->insert(item); });

};


template<int IFORM, typename TM>
void get_surface(TM const &m, Cache<TM> const &cache, IdSet<TM> *surface,
                 bool is_out_boundary = true)
{
    typedef TM mesh_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename mesh_type::vector_type vector_type;

    typedef typename mesh_type::id_type id_type;
    size_t MASK = m.id_mask();

    on_surface(m, cache,
               [&](typename Cache<TM>::value_type const &item)
               {
                   id_type ids_0[mesh_type::MAX_NUM_OF_NEIGHBOURS];
                   id_type ids_1[mesh_type::MAX_NUM_OF_NEIGHBOURS];


                   int num_0 = m.get_adjacent_cells(IFORM, mesh_type::TAG_VOLUME,
                                              item.first + mesh_type::_DA, ids_0);

                   for (int i = 0; i < num_0; ++i)
                   {

                       int num_1 = m.get_adjacent_cells(VERTEX, ids_0[i], ids_1);

                       int count = 0;

                       for (int j = 0; j < num_1; ++j)
                       {
                           typename Cache<TM>::const_accessor acc;

                           bool t_is_out = true;

                           if (cache.find(acc, ids_1[j] & MASK))
                           {
                               t_is_out = (std::get<0>(acc->second) > 0);
                           }

                           if (is_out_boundary == t_is_out)
                           {
                               ++count;
                           }
                       }
                       if (count == num_1)
                       {
                           surface->insert(ids_0[i]);
                       }
                   }
               });

}

template<int IFORM, typename TM, typename TB>
void create_id_set(TM const &m, TB const &box, IdSet<TM> *res)
{
    parallel::parallel_for(m.template make_box_range<IFORM>(box),
                           [&](typename TM::range_type const &r) { for (auto const &s:r) { res->insert(s); }});

}


}} // namespace simpla { namespace model

#endif //SIMPLA_SURFACE_H

