/**
 * @file particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

#include <vector>
#include <list>
#include <map>

#include "../parallel/parallel.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "../gtl/utilities/memory_pool.h"

namespace simpla
{
namespace particle
{


template<typename ...> struct Particle;
template<typename ...> struct ParticleEngine;
template<typename TAGS, typename M> using particle_t= Particle<ParticleEngine<TAGS>, M>;

template<typename P, typename M>
struct Particle<P, M>
        : public P,
          public parallel::concurrent_hash_map<typename M::id_type, std::list<typename P::sample_type>>,
          public std::enable_shared_from_this<Particle<P, M>>
{

private:

    typedef M mesh_type;

    typedef P engine_type;

    typedef Particle<engine_type, mesh_type> this_type;

public:

    static constexpr int iform = VOLUME;

    using typename engine_type::point_type;
    using typename engine_type::sample_type;
    typedef sample_type value_type;
private:

    using engine_type::push;
    using engine_type::project;
    using engine_type::lift;

    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;


    typedef std::list<value_type> bucket_type;

    typedef parallel::concurrent_hash_map <id_type, bucket_type> container_type;

    typedef std::map<id_type, bucket_type> buffer_type;

    static constexpr int ndims = mesh_type::ndims;

    mesh_type const &m_mesh_;

public:


    template<typename ...Args>
    Particle(mesh_type const &m, Args &&...args);

    Particle(this_type const &) = delete;

    virtual ~Particle();

    virtual void deploy();

    virtual void sync() { rehash(); };


    template<typename TDict, typename ...Others>
    void load(TDict const &dict, Others &&...others);

    template<typename OS> OS &print(OS &os) const;

    template<typename TField>
    void integral(id_type const &s, TField *res) const;

    template<typename TField>
    void integral(range_type const &, TField *res) const;

    template<typename TField>
    void integral(TField *res) const;

    template<typename ...Args>
    void push(id_type const &s, Args &&...args);

    template<typename ...Args>
    void push(range_type const &, Args &&...args);

    template<typename ...Args> void push(Args &&...args);


//! @}

//! @ingroup as container bundle @{

    id_type hash(value_type const &p) const { return m_mesh_.id(project(p), mesh_type::node_id(iform)); }


    DataSet dataset() const;

    void dataset(DataSet const &);

    void push_back(value_type const &p);

    template<typename InputIter>
    void push_back(InputIter const &b, InputIter const &e);

    void erase(id_type const &k) { container_type::erase(k); };

    void erase(range_type const &r);

    void erase_all() { container_type::clear(); }

    using container_type::clear;

    //! @}
    //! @{
    size_t size(range_type const &r, container_type const &buffer) const;

    size_t size(range_type const &r) const;

    size_t size() const;

    //! @}
    //! @{

    template<typename OutputIT> OutputIT copy(id_type s, OutputIT out_it) const;

    template<typename OutputIT> OutputIT copy(range_type const &r, OutputIT out_it) const;

    template<typename OutputIT> OutputIT copy(OutputIT out_it) const;

    //! @}
    //! @{

    void merge(id_type const &s, container_type *other);

    void merge(range_type const &r, container_type *other);

    void merge(container_type *other);

    static void merge(buffer_type *buffer, container_type *other);


    //! @}
    //! @{

    void rehash(id_type const &s, container_type *other);

    void rehash(range_type const &r, container_type *other);

    void rehash();

    //! @}


    template<typename TGen, typename ...Args> void generator(id_type s, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(range_type const &, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(TGen &gen, size_t pic, Args &&...args);

    /**
     * @require TConstraint = map<id_type,TARGS>
     *          Fun         = function<void(TARGS const &,sample_type*)>
     */
    template<typename TConstraint, typename TFun> void accept(TConstraint const &, TFun const &fun);

    template<typename TRange, typename Predicate> void remove_if(TRange const &r, Predicate const &pred);


private:
    void sync(container_type const &buffer, parallel::DistributedObject *dist_obj, bool update_ghost = true);

};//class Particle

template<typename P, typename M>
template<typename ...Args>
Particle<P, M>::Particle(M const &m, Args &&...args) :
        engine_type(std::forward<Args>(args)...), m_mesh_(m)
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
//*******************************************************************************

template<typename P, typename M>
template<typename TField>
void Particle<P, M>::integral(id_type const &s, TField *J) const
{
    static constexpr int f_iform = traits::iform<TField>::value;

    auto x0 = m_mesh_.point(s);


    id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = m_mesh_.get_adjacent_cells(iform, s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (container_type::find(acc1, neighbours[i]))
        {
            for (auto const &p:acc1->second)
            {
                typename ::simpla::traits::field_value_type<TField>::type v;
                engine_type::integral(x0, p, &v);
                (*J)[s] += m_mesh_.template sample<f_iform>(s, v);
            }
        }
    }
};

template<typename P, typename M>
template<typename TField>
void Particle<P, M>::integral(range_type const &r, TField *J) const
{
    // TODO cache J, base on r
    for (auto const &s:r) { integral(s, J); }
};

template<typename P, typename M>
template<typename TField>
void Particle<P, M>::integral(TField *J) const
{

    static constexpr int f_iform = traits::iform<TField>::value;
    m_mesh_.template for_each_boundary<f_iform>([&](range_type const &r) { integral(r, J); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    m_mesh_.template for_each_center<f_iform>([&](range_type const &r) { integral(r, J); });

    dist_obj.wait();

}
//*******************************************************************************

template<typename P, typename M>
template<typename ...Args>
void Particle<P, M>::push(id_type const &s, Args &&...args)
{
    typename container_type::accessor acc;

    if (container_type::find(acc, s))
    {
        for (auto &p:acc->second)
        {
            engine_type::push(&p, std::forward<Args>(args)...);
        }
    }


};

template<typename P, typename M>
template<typename ...Args>
void Particle<P, M>::push(range_type const &r, Args &&...args)
{
    // TODO cache args, base on s or r
    for (auto const &s:r) { push(s, std::forward<Args>(args)...); }
};

template<typename P, typename M> template<typename ...Args>
void Particle<P, M>::push(Args &&...args)
{
    // @note this is lock free

    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_mesh_.template for_each_boundary<iform>([&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_mesh_.template for_each_center<iform>([&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    rehash();
}


//**************************************************************************************************

template<typename P, typename M>
size_t Particle<P, M>::size(range_type const &r, container_type const &c) const
{

    return parallel::parallel_reduce(r, 0,
                                     [&](range_type &r, size_t init) -> size_t
                                     {
                                         for (auto const &s:r)
                                         {
                                             typename container_type::accessor acc;
                                             if (c.find(acc, s))
                                                 init += acc->second.size();
                                         }

                                         return init;
                                     },
                                     [](size_t x, size_t y) -> size_t
                                     {
                                         return x + y;
                                     }
    );

}

template<typename P, typename M>
size_t Particle<P, M>::size(range_type const &r) const
{
    return size(r, *this);
}

template<typename P, typename M>
size_t Particle<P, M>::size() const
{
    return size(m_mesh_.template range<iform>(), *this);
}


//**************************************************************************************************

template<typename P, typename M>
void Particle<P, M>::erase(range_type const &r)
{
    parallel::parallel_for(r, [&](range_type const &r) { for (auto const &s:r) { container_type::erase(s); }});

}
//**************************************************************************************************



template<typename P, typename M>
template<typename OutputIter>
OutputIter Particle<P, M>::copy(id_type s, OutputIter out_it) const
{
    typename container_type::const_accessor c_accessor;
    if (container_type::find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename P, typename M>
template<typename OutputIter>
OutputIter Particle<P, M>::copy(range_type const &r, OutputIter out_it) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy(s, out_it); }
    return out_it;
}

template<typename P, typename M>
template<typename OutputIter>
OutputIter Particle<P, M>::copy(OutputIter out_it) const
{
    return copy(m_mesh_.template range<iform>(), out_it);
}
//*******************************************************************************


template<typename P, typename M>
void Particle<P, M>::merge(id_type const &s, container_type *buffer)
{
    typename container_type::accessor acc0;
    typename container_type::accessor acc1;

    if (buffer->find(acc1, s))
    {
        container_type::insert(acc0, s);
        acc0->second.splice(acc0->second.end(), acc1->second);
    }
}

template<typename P, typename M>
void Particle<P, M>::merge(buffer_type *buffer, container_type *other)
{
    for (auto &item:*buffer)
    {
        typename container_type::accessor acc1;

        other->insert(acc1, item.first);

        acc1->second.splice(acc1->second.end(), item.second);
    }
};

template<typename P, typename M>
void Particle<P, M>::merge(range_type const &r, container_type *other)
{
    for (auto const &s:r) { merge(s, other); }
}


template<typename P, typename M>
void Particle<P, M>::merge(container_type *other)
{
    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { merge(r, other); });

    m_mesh_.template for_each_boundary<iform>([&](range_type const &r) { merge(r, other); });

    m_mesh_.template for_each_center<iform>([&](range_type const &r) { merge(r, other); });
}
//*******************************************************************************

template<typename P, typename M>
void Particle<P, M>::sync(container_type const &buffer, parallel::DistributedObject *dist_obj, bool update_ghost)
{

    DataType d_type = traits::datatype<value_type>::create();

    typename mesh_type::index_tuple memory_min, memory_max;
    typename mesh_type::index_tuple local_min, local_max;

    std::tie(memory_min, memory_max) = m_mesh_.memory_index_box();
    std::tie(local_min, local_max) = m_mesh_.local_index_box();


    for (unsigned int tag = 0, tag_e = (1U << (m_mesh_.ndims * 2)); tag < tag_e; ++tag)
    {
        nTuple<int, 3> coord_offset;

        bool tag_is_valid = true;

        index_tuple send_min, send_max;

        send_min = local_min;
        send_max = local_max;

        for (int n = 0; n < ndims; ++n)
        {
            if (((tag >> (n * 2)) & 3UL) == 3UL)
            {
                tag_is_valid = false;
                break;
            }

            coord_offset[n] = ((tag >> (n * 2)) & 3U) - 1;
            if (update_ghost)
            {
                switch (coord_offset[n])
                {
                    case 0:
                        break;
                    case -1: //left
                        send_min[n] = memory_min[n];
                        send_max[n] = local_min[n];
                        break;
                    case 1: //right
                        send_min[n] = local_max[n];
                        send_max[n] = memory_max[n];
                        break;
                    default:
                        tag_is_valid = false;
                        break;
                }
            }
            else
            {
                switch (coord_offset[n])
                {
                    case 0:
                        break;
                    case -1: //left
                        send_min[n] = local_min[n];
                        send_max[n] = local_min[n] + local_min[n] - memory_min[n];
                        break;
                    case 1: //right
                        send_min[n] = local_max[n] - (memory_max[n] - local_max[n]);
                        send_max[n] = local_max[n];
                        break;
                    default:
                        tag_is_valid = false;
                        break;
                }
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
                std::shared_ptr<void> p_send(nullptr), p_recv(nullptr);

                auto send_range = m_mesh_.template make_range<iform>(send_min, send_max);

                size_t send_size = size(send_range, buffer);

                p_send = sp_alloc_memory(send_size * sizeof(value_type));

                auto p = reinterpret_cast<value_type *>( p_send.get());

                for (auto const &s:send_range)
                {
                    typename container_type::accessor acc;

                    if (buffer.find(acc, s))
                    {
                        p = std::copy(acc->second.begin(), acc->second.end(), p);
                    }

                }

                dist_obj->add_link_send(coord_offset, d_type, p_send, send_size);

                dist_obj->add_link_recv(coord_offset, d_type);

            }
            catch (std::exception const &error)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("add coommnication link error", error.what());

            }
        }
    }
}


template<typename P, typename M>
void Particle<P, M>::rehash(id_type const &key, container_type *other)
{
    typename container_type::accessor acc0;

    if (container_type::find(acc0, key))
    {
        buffer_type buffer;

        auto &src = acc0->second;

        auto it = src.begin(), ie = src.end();

        while (it != ie)
        {
            auto p = it;

            ++it;

            auto dest = hash(*p);

            if (dest != key) { buffer[dest].splice(buffer[dest].end(), src, p); }
        }

        merge(&buffer, other);

        {
            typename container_type::accessor acc1;
            other->insert(acc1, key);
            acc1->second.splice(acc1->second.end(), src);
        }
    }
}

template<typename P, typename M>
void Particle<P, M>::rehash(range_type const &r, container_type *buffer)
{
    for (auto const &s:r) { rehash(s, buffer); }
}

template<typename P, typename M>
void Particle<P, M>::rehash()
{

    container_type buffer;
    /**
     *  move particle out of cell[s]
     *  ***************************
     *  *     ............        *
     *  *     .   center .        *
     *  *     .+boundary===> ghost*
     *  *     ............        *
     *  ***************************
     */
    m_mesh_.template for_each_boundary<iform>([&](range_type const &r) { rehash(r, &buffer); });


    //**************************************************************************************
    // sync ghost area in buffer

    parallel::DistributedObject dist_obj;

    sync(buffer, &dist_obj);
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
    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { rehash(r, &buffer); });
    /**
     *
     *  ***************************
     *  *     ............        *
     *  *     .  center  .        *
     *  *     .  <===>   .        *
     *  *     ............        *
     *  ***************************
     */
    m_mesh_.template for_each_center<iform>([&](range_type const &r) { rehash(r, &buffer); });


    //collect moved particle
    m_mesh_.template for_each_center<iform>([&](range_type const &r) { merge(r, &buffer); });

    m_mesh_.template for_each_boundary<iform>([&](range_type const &r) { merge(r, &buffer); });

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

    for (auto const &item :  dist_obj.recv_buffer)
    {
        value_type const *p = reinterpret_cast<value_type const *>(std::get<1>(item).data.get());
        push_back(p, p + std::get<1>(item).memory_space.size());
    }
}

template<typename P, typename M>
void Particle<P, M>::push_back(value_type const &p)
{
    typename container_type::accessor acc;
    container_type::insert(acc, hash(p));
    acc->second.push_back(p);
}

template<typename P, typename M>
template<typename InputIt>
void Particle<P, M>::push_back(InputIt const &b, InputIt const &e)
{
    // fixme need parallize
    std::map<id_type, bucket_type> buffer;
    for (auto it = b; it != e; ++it)
    {
        buffer[hash(*it)].push_back(*it);
    }
    merge(&buffer, this);
}


template<typename P, typename M>
DataSet Particle<P, M>::dataset() const
{

    auto r = m_mesh_.template range<iform>();
    size_t count = static_cast<int>(size(r));

    auto data = sp_alloc_memory(count * sizeof(value_type));

    copy(r, reinterpret_cast<value_type *>( data.get()));

    DataSet ds = traits::make_dataset(traits::datatype<value_type>::create(), data, count);

//    ds.properties.append(engine_type::properties);

    return std::move(ds);
}

template<typename P, typename M>
void Particle<P, M>::dataset(DataSet const &ds)
{
    size_t count = ds.memory_space.size();

    value_type const *p = reinterpret_cast<value_type const *>(ds.data.get());

    push_back(p, p + count);

//    engine_type::properties.append(ds.properties);

    rehash();
}


template<typename P, typename M>
template<typename TGen, typename ...Args>
void Particle<P, M>::generator(id_type s, TGen &gen, size_t pic, Args &&...args)
{
    auto g = gen.generator(pic, m_mesh_.volume(s), m_mesh_.box(s),
                           std::forward<Args>(args)...);

    typename container_type::accessor acc;
    container_type::insert(acc, s);
    std::copy(std::get<0>(g), std::get<1>(g), std::back_inserter(acc->second));
}


template<typename P, typename M>
template<typename TGen, typename ...Args>
void Particle<P, M>::generator(const range_type &r, TGen &gen, size_t pic, Args &&...args)
{
    for (auto const &s:r) { generator(s, gen, pic, std::forward<Args>(args)...); }
}


template<typename P, typename M>
template<typename TGen, typename ...Args>
void Particle<P, M>::generator(TGen &gen, size_t pic, Args &&...args)
{
//    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) {
// generator(r, std::forward<Args>(args)...); });

    size_t num_of_particle = m_mesh_.template range<iform>().size() * pic;

    gen.reserve(num_of_particle);


    m_mesh_.template for_each_boundary<iform>(
            [&](range_type const &r) { generator(r, gen, pic, std::forward<Args>(args)...); });

    parallel::DistributedObject dist_obj;

    sync(*this, &dist_obj, false);

    dist_obj.sync();

    m_mesh_.template for_each_center<iform>(
            [&](range_type const &r) { generator(r, gen, pic, std::forward<Args>(args)...); });

    dist_obj.wait();

    for (auto const &item :  dist_obj.recv_buffer)
    {
        value_type const *p = reinterpret_cast<value_type const *>(std::get<1>(item).data.get());
        push_back(p, p + std::get<1>(item).memory_space.size());
    }
}

template<typename P, typename M>
template<typename TRange, typename Predicate>
void Particle<P, M>::remove_if(TRange const &r, Predicate const &pred)
{
    parallel::parallel_for(r,
                           [&](TRange const &r)
                           {
                               for (auto const &s:r)
                               {
                                   typename container_type::accessor acc;

                                   if (container_type::find(acc, std::get<0>(s)))
                                   {
                                       acc->second.remove_if(
                                               [&](value_type const &p)
                                               {
                                                   return pred(s, p);
                                               });
                                   }


                               }
                           }

    );
}

template<typename P, typename M>
template<typename TConstraint, typename TFun>
void Particle<P, M>::accept(TConstraint const &constraint, TFun const &fun)
{
    container_type buffer;
    parallel::parallel_for(
            constraint.range(),
            [&](typename TConstraint::range_type const &r)
            {
                for (auto const &item:r)
                {

                    typename container_type::accessor acc;

                    if (container_type::find(acc, std::get<0>(item)))
                    {
                        for (auto &p:acc->second)
                        {
                            fun(item, &p);
                        }
                    }

                    rehash(std::get<0>(item), &buffer);
                }
            }

    );
    merge(&buffer);
}
}//{namespace particle

namespace traits
{
template<typename P, typename M>
struct iform<particle::Particle<P, M>> :
        public std::integral_constant<int, particle::Particle<P, M>::iform>
{
};

template<typename P, typename M>
struct value_type<particle::Particle<P, M>>
{
    typedef typename particle::Particle<P, M>::sample_type type;
};


} //namespace traits
}//namespace simpla


#endif /* CORE_PARTICLE_PARTICLE_H_ */
