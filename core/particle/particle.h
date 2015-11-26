/**
 * @file particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

#include <vector>
#include "../gtl/containers/unordered_set.h"

namespace simpla
{
template<typename...> struct Particle;


template<typename PEngine, typename M>
struct Particle<PEngine, M>
        : public PEngine,
          public UnorderedSet<typename PEngine::point_type, typename M::id_type>,
          public std::enable_shared_from_this<Particle<PEngine, M>>
{
    static constexpr int iform = VOLUME;

private:

    typedef M mesh_type;

    typedef PEngine engine_type;

    typedef Particle<engine_type, mesh_type> this_type;

    using typename engine_type::point_type;

    using engine_type::push;
    using engine_type::project;
    using engine_type::lift;
    using engine_type::function_value;

    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;
    static constexpr int ndims = mesh_type::ndims;

    mesh_type const &m_mesh_;

public:

    typedef UnorderedSet<point_type, id_type> storage_type;

    template<typename ...Args>
    Particle(mesh_type const &m, Args &&...args);

    Particle(this_type const &);

    virtual ~Particle();

    virtual void deploy();

    virtual void rehash();

    virtual void sync() { rehash(); };

    virtual DataSet dataset() const;

    virtual void dataset(DataSet const &);

    template<typename TDict, typename ...Others>
    void load(TDict const &dict, Others &&...others);

    template<typename OS> OS &print(OS &os) const;


    template<typename TField>
    void integral(range_type const &, TField *res) const;

    template<typename TField>
    void integral(TField *res) const;


    template<typename ...Args>
    void push(range_type const &, Args &&...args);

    template<typename ...Args>
    void push(Args &&...args);

    void insert(point_type const &p)
    {
        storage_type::insert(m_mesh_.hash(project(p)), p);
    }
//! @}

};//class Particle
namespace traits
{
template<typename PEngine, typename M>
struct iform<Particle<PEngine, M>> :
        public std::integral_constant<int, Particle<PEngine, M>::iform>
{
};

template<typename PEngine, typename M>
struct value_type<Particle<PEngine, M>>
{
    typedef typename Particle<PEngine, M>::point_type type;
};


}

//namespace traits
template<typename P, typename M>
template<typename ...Args>
Particle<P, M>::Particle(M const &m, Args &&...args) :
        engine_type(std::forward<Args>(args)...), storage_type(), m_mesh_(m)
{
}

template<typename P, typename M>
Particle<P, M>::Particle(Particle<P, M> const &other) :
        engine_type(other), storage_type(other), m_mesh_(other.m_mesh_)
{
}

template<typename P, typename M>
Particle<P, M>::~Particle()
{
}


template<typename P, typename M>
template<typename TDict, typename ...Others>
void Particle<P, M>::load(TDict const &dict, Others &&...others)
{
    engine_type::load(dict, std::forward<Others>(others)...);
}

template<typename P, typename M>
template<typename OS>
OS &Particle<P, M>::print(OS &os) const
{
    engine_type::print(os);
    return os;
}

template<typename P, typename M>
void Particle<P, M>::deploy()
{
}

template<typename P, typename M>
template<typename TField>
void Particle<P, M>::integral(range_type const &r, TField *J) const
{
    // TODO cache J, base on r
    for (auto const &s:r)
    {
        static constexpr int MAX_NEIGHBOUR_NUM = 12;
        id_type neighbour[MAX_NEIGHBOUR_NUM];
        auto x0 = m_mesh_.point(s);
        int num = m_mesh_.get_neighbour(s, iform);

        for (int i = 0; i < num; ++i)
            for (auto const &p:(*this)[neighbour[i]])
            {
                (*J)[s] += m_mesh_.RBF(project(p), x0) *
                           m_mesh_.sample(s, engine_type::integral_v(p));
            }
    }
};

template<typename P, typename M>
template<typename TField>
void Particle<P, M>::integral(TField *J) const
{
    // @note this is lock free

    static constexpr int f_iform = traits::iform<TField>::value;
    m_mesh_.template for_each_boundary<f_iform>([&](range_type const &r) { integral(r, J); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    m_mesh_.template for_each_center<f_iform>([&](range_type const &r) { integral(r, J); });

    dist_obj.wait();

}

template<typename P, typename M>
template<typename ...Args>
void Particle<P, M>::push(range_type const &r, Args &&...args)
{
    // TODO cache args, base on s or r
    for (auto const &s:r)
    {
        for (auto &p:(*this)[s])
        {
            engine_type::push(&p, std::forward<Args>(args)...);
        }
    }
};

template<typename P, typename M> template<typename ...Args>
void Particle<P, M>::push(Args &&...args)
{
    // @note this is lock free

    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_mesh_.template for_each<iform>([&](range_type const &r) { push(r, std::forward<Args>(args)...); });
}


template<typename P, typename M>
void Particle<P, M>::rehash()
{

    storage_type buffer;
    /**
     *  move particle out of cell[s]
     *  ***************************
     *  *     ............        *
     *  *     .   center .        *
     *  *     .+boundary===> ghost*
     *  *     ............        *
     *  ***************************
     */
    m_mesh_.template for_each_boundary<iform>(
            [&](range_type const &r) { /* FIXME storage_type::rehash(r, &buffer); */ });


    //**************************************************************************************
    // sync ghost area in buffer



    parallel::DistributedObject dist_obj;

    auto d_type = traits::datatype<point_type>::create();

    typename mesh_type::index_tuple memory_min, memory_max;

    std::tie(memory_min, memory_max) = m_mesh_.memory_index_box();


    std::vector<std::tuple<size_t, std::shared_ptr<void>>> send_buffer;
    std::vector<std::tuple<size_t, std::shared_ptr<void>>> recv_buffer;

    for (unsigned int tag = 0, tag_e = (1U << (m_mesh_.ndims * 2)); tag < tag_e; ++tag)
    {
        nTuple<int, 3> coord_offset;

        bool tag_is_valid = true;

        index_tuple send_min, send_max;

        std::tie(send_min, send_max) = m_mesh_.local_index_box();

        for (int n = 0; n < ndims; ++n)
        {
            if (((tag >> (n * 2)) & 3UL) == 3UL)
            {
                tag_is_valid = false;
                break;
            }

            coord_offset[n] = ((tag >> (n * 2)) & 3U) - 1;

            switch (coord_offset[n])
            {
                case 0:
                    break;
                case -1: //left
                    send_max[n] = send_min[n];
                    send_min[n] = memory_min[n];
                    break;
                case 1: //right
                    send_min[n] = send_max[n];
                    send_max[n] = memory_max[n];
                    break;
                default:
                    tag_is_valid = false;
                    break;
            }

            if (send_max[n] == send_min[n])
            {
                tag_is_valid = false;
                break;
            }

        }

        if (tag_is_valid && (coord_offset[0] != 0 || coord_offset[1] != 0 || coord_offset[2] != 0))
        {
            try
            {
                std::shared_ptr<void> p_send, p_recv;

                size_t send_size = 0;// FIXME storage_type::copy(m_mesh_.make_range<iform>(send_min, send_max), &p_send);

                send_buffer.push_back(std::make_tuple(send_size, p_send));

                recv_buffer.push_back(std::make_tuple(0, p_recv));

                dist_obj.add_link_send(&coord_offset[0], send_size, d_type, &p_send);

                dist_obj.add_link_recv(&coord_offset[0], 0, d_type, &p_recv);


            }
            catch (std::exception const &error)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("add coommnication link error", error.what());

            }
        }

    }


    dist_obj.sync();

    //**************************************************************************************

    //
    /**
     *  collect particle from   ghost ->boundary
     *  ***************************
     *  *     ............        *
     *  *     .   center .        *
     *  *     .+boundary<== ghost *
     *  *     ............        *
     *  ***************************
     */
    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { /* fixme storage_type::rehash(r, &buffer); */});
    /**
     *
     *  ***************************
     *  *     ............        *
     *  *     .  center  .        *
     *  *     .  <===>   .        *
     *  *     ............        *
     *  ***************************
     */
    m_mesh_.template for_each_center<iform>(
            [&](range_type const &r) {/* fixme  storage_type::rehash(r, &buffer); */ });


    //collect moved particle
    m_mesh_.template for_each_center<iform>(
            [&](range_type const &r) { for (auto const &s:r) {/* fixme(*this)[s].merge(buffer[s]); */}});
    m_mesh_.template for_each_boundary<iform>(
            [&](range_type const &r) { for (auto const &s:r) {/* fixme(*this)[s].merge(buffer[s]); */}});
    // wait

    dist_obj.wait();
    /**
     *
     *  ***************************
     *  *     ............        *
     *  *     .  center  .        *
     *  *     .          . ghost <== neighbour
     *  *     ............        *
     *  ***************************
     */

    for (auto const &item : recv_buffer)
    {
        point_type *p = reinterpret_cast<point_type *>(std::get<1>(item).get());
//        storage_type::insert(p, p + std::get<0>(item));
    }
}


template<typename P, typename M>
DataSet Particle<P, M>::dataset() const
{
    DataSet ds;

    size_t count = 0;//fixme storage_type::raw_copy(m_mesh_.template make_range<iform>(), &ds.data);

    size_t offset, total_count;

    std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, count);

    ds.dataspace = DataSpace(1, &total_count);

    ds.dataspace.select_hyperslab(&offset, nullptr, &count, nullptr);

    ds.memory_space = DataSpace(1, &count);

    // fixme ds.properties.append(engine_type::properties);

    return std::move(ds);
}

template<typename P, typename M>
void Particle<P, M>::dataset(DataSet const &ds)
{
    size_t count = ds.memory_space.size();

    point_type const *p = reinterpret_cast<point_type *>(ds.data.get());

    //fixme    storage_type::insert(p, p + count);

}


}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_H_ */
