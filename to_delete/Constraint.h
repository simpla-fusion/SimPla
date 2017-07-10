/**
 * @file Constraint.h
 * @author salmon
 * @date 2015-11-30.
 */

#ifndef SIMPLA_CONSTRAINT_H
#define SIMPLA_CONSTRAINT_H

#include <simpla/utilities/Parallel.h>
#include <simpla/manifold/ManifoldTraits.h>

namespace simpla { namespace geometry
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

template<typename TM> using Surface=Constraint<TM, Real, typename TM::point_type>;

template<typename TM> using CellCache = Surface<TM>;

/**
 *  id = 0  union
 *         1  intersection
 *       -1  Difference
 */

template<typename TM, int IFORM = VERTEX>
void update_cache(TM const &m, geometry::GeoObject const &geo, CellCache<TM> *cache, int flag = 0)
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
                                   size_t id;
                                   Real d = geo.nearest_point(&x);
                                   typename CellCache<TM>::value_type item(s, std::make_tuple(d, x));
                                   typename CellCache<TM>::accessor acc;
                                   bool cond = cache->insert(acc, s);
                                   if (!cond)
                                   {
                                       switch (flag)
                                       {

                                           case -1: //difference
                                               cond = -d > std::get<0>(acc->second);
                                           case 2: // intersection
                                               cond = d > std::get<0>(acc->second);
                                           default: //union
                                               cond = d < std::get<0>(acc->second);
                                       }
                                   };
                                   if (cond)
                                   {
                                       std::get<0>(acc->second) = d;
                                       std::get<1>(acc->second) = x;
                                   }
                               }
                           }

    );


}

enum
{
    INSIDE = 1, OUTSIDE = 2, ON_SURFACE = 4
};

/**
 *  id < 0 out of surface
 *       = 0 on surface
 *       > 0 in surface
 */
template<typename TM, typename TRange, typename Func>
void search_cache(TM const &m, TRange const &r0, CellCache<TM> const &cache, int flag, Func const &func)
{

    typedef TM mesh_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename mesh_type::vector_type vector_type;

    typedef typename mesh_type::id_type id_type;


    size_t MASK = m.periodic_id_mask();

    parallel::parallel_for(
            r0,
            [&](TRange const &r)
            {
                for (auto const &v_s: r)
                {

                    id_type p[mesh_type::MAX_NUM_OF_NEIGHBOURS];

                    int num = m.get_adjacent_cells(VERTEX, v_s, p);

                    int count = 0;

                    for (int i = 0; i < num; ++i)
                    {
                        typename CellCache<TM>::const_accessor acc;

                        if (cache.find(acc, (p[i] & MASK)))
                        {
                            if (std::get<0>(acc->second) > 0) { ++count; }
                        }
                    }
                    if (
                            ((count == num) && ((flag & INSIDE) > 0)) ||
                            ((count == 0) && ((flag & OUTSIDE) > 0)) ||
                            ((count > 0 && count < num) && ((flag & ON_SURFACE) > 0))
                            )
                    {
                        func(v_s);
                    }

                }

            }

    );


}


template<typename TM, typename ...Args>
void get_cell_on_surface(TM const &m, geometry::GeoObject const &geo, Args &&...args)
{

    CellCache<TM> cache;

    update_cache(geo, m, &cache);

    get_cell_on_surface(cache, cache, std::forward<Args>(args)...);
}


template<typename TM, typename TRange>
void get_cell_on_surface(TM const &m, TRange const &r0, CellCache<TM> const &cache, Surface<TM> *surface)
{
    typedef TM mesh_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename mesh_type::vector_type vector_type;

    typedef typename mesh_type::id_type id_type;


    size_t MASK = m.periodic_id_mask();

    search_cache(m, r0, cache, ON_SURFACE,
                 [&](typename TM::id_type const &s)
                 {
                     typename CellCache<TM>::const_accessor acc;

                     if (cache.find(acc, (((s | mesh_type::FULL_OVERFLOW_FLAG) - mesh_type::_DA) & MASK)))
                     {
                         surface->insert(*acc);
                     }

                 });

};

template<typename TM>
void get_cell_on_surface(TM const &m, CellCache<TM> const &cache, Surface<TM> *surface)
{
    get_cell_on_surface(m, m.template range<VOLUME>(), cache, surface);
}

template<int IFORM, typename TRange, typename TM>
void get_cell_on_surface(TM const &m, TRange const &r0, CellCache<TM> const &cache, IdSet<TM> *surface,
                         bool is_out_boundary = true)
{
    typedef TM mesh_type;

    typedef typename mesh_type::point_type point_type;

    typedef typename mesh_type::vector_type vector_type;

    typedef typename mesh_type::id_type id_type;

    size_t MASK = m.periodic_id_mask();

    search_cache(m, r0, cache, ON_SURFACE,
                 [&](typename TM::id_type const &v_s)
                 {
                     id_type ids_0[mesh_type::MAX_NUM_OF_NEIGHBOURS];
                     id_type ids_1[mesh_type::MAX_NUM_OF_NEIGHBOURS];


                     int num_0 = m.get_adjacent_cells(IFORM, mesh_type::TAG_VOLUME, v_s, ids_0);

                     for (int i = 0; i < num_0; ++i)
                     {

                         int num_1 = m.get_adjacent_cells(VERTEX, ids_0[i], ids_1);

                         int count = 0;

                         for (int j = 0; j < num_1; ++j)
                         {
                             typename CellCache<TM>::const_accessor acc;

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

template<int IFORM, typename TM>
void get_cell_on_surface(TM const &m, CellCache<TM> const &cache, IdSet<TM> *surface,
                         bool is_out_boundary = true)
{
    get_cell_on_surface<IFORM>(m, m.template range<VOLUME>(), cache, surface, is_out_boundary);
};


template<typename TM, typename TRange>
void get_cell_in_surface(TM const &m, TRange const &r0, CellCache<TM> const &cache, IdSet<TM> *res)
{
    search_cache(m, r0, cache, INSIDE, [&](typename TM::id_type const &s) { res->insert(s); });
};

template<int IFORM, typename TM>
void get_cell_in_surface(TM const &m, CellCache<TM> const &cache, IdSet<TM> *res)
{
    search_cache(m, m.template range<IFORM>(), cache, INSIDE, [&](typename TM::id_type const &s) { res->insert(s); });
};


template<typename TM, typename TRange>
void get_cell_out_surface(TM const &m, TRange const &r0, CellCache<TM> const &cache, IdSet<TM> *res)
{
    search_cache(m, r0, cache, OUTSIDE, [&](typename TM::id_type const &s) { res->insert(s); });
};

template<int IFORM, typename TM>
void get_cell_out_surface(TM const &m, CellCache<TM> const &cache, IdSet<TM> *res)
{
    search_cache(m, m.template range<IFORM>(), cache, OUTSIDE, [&](typename TM::id_type const &s) { res->insert(s); });
};


template<typename TM, typename TRange, typename TSet>
void create_id_set(TM const &m, TRange const &r0, TSet *res)
{
    size_t MASK = m.id_mask();
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { res->insert(s & MASK); }});

}


}} // namespace simpla { namespace geometry

#endif //SIMPLA_CONSTRAINT_H

