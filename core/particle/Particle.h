/**
 * @file Particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

#include <vector>
#include <list>
#include <map>

#include "../parallel/Parallel.h"
#include "../gtl/design_pattern/singleton_holder.h"
#include "../gtl/utilities/memory_pool.h"

namespace simpla { namespace particle
{
template<typename ...> struct Particle;
template<typename ...> struct ParticleEngine;
template<typename TAGS, typename M> using particle_t= Particle<ParticleEngine<TAGS>, M>;

template<typename P, typename M, typename ...Policies>
struct Particle<P, M, Policies...> : public P, public Policies ...
{

public:
    typedef M mesh_type;
    typedef P engine_type;

    typedef Particle<P, M, Policies...> this_type;

//    typedef typename this_type::calculus_policy calculus_policy;

    typedef typename this_type::interpolate_policy interpolate_policy;

    typedef typename engine_type::point_type point_type;
    typedef typename engine_type::sample_type sample_type;
    typedef sample_type value_type;

    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;

    typedef typename M::AttributeEntity base_type;

    typedef std::list<value_type> bucket_type;

    typedef parallel::concurrent_hash_map <id_type, bucket_type> container_type;

    std::shared_ptr<container_type> m_data_;
    mesh_type const *m_mesh_;
    //   parallel::concurrent_hash_map<typename M::id_type, std::list<typename P::sample_type>>

public:

    static constexpr int iform = VOLUME;


    typedef std::map<id_type, bucket_type> buffer_type;

    static constexpr int ndims = mesh_type::ndims;


public:


    Particle(mesh_type const &m);

    Particle(this_type const &);

    Particle(this_type &&);

    this_type &operator=(this_type const &other);

    void swap(this_type const &other);

    virtual ~Particle();

    virtual void deploy();

    virtual void clear();

    virtual void sync();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    template<typename TDict> void load(TDict const &dict);


    constexpr id_type hash(value_type const &p) const
    {
        return m_mesh_->id(engine_type::project(p), mesh_type::node_id(iform));
    }

    template<typename TGen, typename ...Args> void generator(id_type s, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(range_type const &, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(TGen &gen, size_t pic, Args &&...args);

    /**
     * @require TConstraint = map<id_type,TARGS>
     *          Fun         = function<void(TARGS const &,sample_type*)>
     */
    template<typename TConstraint, typename TFun> void accept(TConstraint const &, TFun const &fun);

    template<typename TRange, typename Predicate> void remove_if(TRange const &r, Predicate const &pred);

    //! @{
    data_model::DataSet data_set() const;

    void data_set(data_model::DataSet const &);

    void push_back(value_type const &p);

    template<typename InputIteratorerator>
    void push_back(InputIteratorerator const &b, InputIteratorerator const &e);

    void erase(id_type const &k) { m_data_->erase(k); };

    void erase(range_type const &r);

    void erase_all() { m_data_->clear(); }


    size_t size(range_type const &r, container_type const &buffer) const;

    size_t size(range_type const &r) const;

    size_t size() const;


    template<typename OutputIT> OutputIT copy(id_type s, OutputIT out_it) const;

    template<typename OutputIT> OutputIT copy(range_type const &r, OutputIT out_it) const;

    template<typename OutputIT> OutputIT copy(OutputIT out_it) const;


    void merge(id_type const &s, container_type *other);

    void merge(range_type const &r, container_type *other);

    void merge(container_type *other);

    static void merge(buffer_type *buffer, container_type *other);


    void rehash(id_type const &s, container_type *other);

    void rehash(range_type const &r, container_type *other);

    void rehash();

    //! @}


    //! @name as container
    //! @{
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


private:
    void sync(container_type const &buffer, parallel::DistributedObject *dist_obj, bool update_ghost = true);

};//class Particle

template<typename P, typename M, typename ...Policies>
Particle<P, M, Policies...>::Particle(M const &m)
        : m_mesh_(&m), m_data_(new container_type) { }

template<typename P, typename M, typename ...Policies>
Particle<P, M, Policies...>::Particle(Particle const &other)
        : m_mesh_(other.m), m_data_(other.m_data_) { }

template<typename P, typename M, typename ...Policies>
Particle<P, M, Policies...>::Particle(Particle &&other)
        : m_mesh_(other.m), m_data_(other.m_data_) { }

template<typename P, typename M, typename ...Policies>
Particle<P, M, Policies...>::~Particle() { }

template<typename P, typename M, typename ...Policies> Particle<P, M, Policies...> &
Particle<P, M, Policies ...>::operator=(const Particle<P, M, Policies...> &other)
{
    Particle<P, M, Policies ...>(other).swap(*this);
    return *this;
}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::swap(const Particle<P, M, Policies...> &other)
{
    std::swap(m_mesh_);
    std::swap(m_data_);
}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::clear()
{
    deploy();
    m_data_->clear();
}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::deploy()
{

}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::sync() { rehash(); };

template<typename P, typename M, typename ...Policies>
template<typename TDict> void
Particle<P, M, Policies...>::load(TDict const &dict) { engine_type::load(dict); }

template<typename P, typename M, typename ...Policies>
std::ostream &Particle<P, M, Policies...>::print(std::ostream &os, int indent) const
{
    return engine_type::print(os, indent + 1);
}


//*******************************************************************************

template<typename P, typename M, typename ...Policies>
template<typename TField> void
Particle<P, M, Policies...>::integral(id_type const &s, TField *J) const
{
    static constexpr int f_iform = traits::iform<TField>::value;

    auto x0 = m_mesh_->point(s);


    id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = m_mesh_->get_adjacent_cells(iform, s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (m_data_->find(acc1, neighbours[i]))
        {
            for (auto const &p:acc1->second)
            {
                typename ::simpla::traits::field_value_type<TField>::type v;

                engine_type::integral(x0, p, &v);

                (*J)[s] += interpolate_policy::template sample<f_iform>(*m_mesh_, s, v);
            }
        }
    }
};

template<typename P, typename M, typename ...Policies>
template<typename TField> void
Particle<P, M, Policies...>::integral(range_type const &r, TField *J) const
{
    // TODO cache J, Base on r
    for (auto const &s:r) { integral(s, J); }
};

template<typename P, typename M, typename ...Policies>
template<typename TField> void
Particle<P, M, Policies...>::integral(TField *J) const
{

    static constexpr int f_iform = traits::iform<TField>::value;
    m_mesh_->template for_each_boundary<f_iform>([&](range_type const &r) { integral(r, J); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    m_mesh_->template for_each_center<f_iform>([&](range_type const &r) { integral(r, J); });

    dist_obj.wait();

}
//*******************************************************************************

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
Particle<P, M, Policies...>::push(id_type const &s, Args &&...args)
{
    typename container_type::accessor acc;

    if (m_data_->find(acc, s))
    {
        for (auto &p:acc->second)
        {
            engine_type::push(&p, std::forward<Args>(args)...);
        }
    }


};

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
Particle<P, M, Policies...>::push(range_type const &r, Args &&...args)
{
    // TODO cache args, Base on s or r
    for (auto const &s:r) { push(s, std::forward<Args>(args)...); }
};

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
Particle<P, M, Policies...>::push(Args &&...args)
{
    // @note this is lock free

    m_mesh_->template for_each_ghost<iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_mesh_->template for_each_boundary<iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    m_mesh_->template for_each_center<iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    rehash();
}


//**************************************************************************************************

template<typename P, typename M, typename ...Policies> size_t
Particle<P, M, Policies...>::size(range_type const &r, container_type const &c) const
{

    return parallel::parallel_reduce(
            r, 0,
            [&](range_type &r, size_t init) -> size_t
            {
                for (auto const &s:r)
                {
                    typename container_type::accessor acc;

                    if (c.find(acc, s)) init += acc->second.size();
                }

                return init;
            },
            [](size_t x, size_t y) -> size_t
            {
                return x + y;
            }
    );

}

template<typename P, typename M, typename ...Policies> size_t
Particle<P, M, Policies...>::size(range_type const &r) const
{
    return size(r, *m_data_);
}

template<typename P, typename M, typename ...Policies> size_t
Particle<P, M, Policies...>::size() const
{
    return size(m_mesh_->template range<iform>(), *m_data_);
}


//**************************************************************************************************

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::erase(range_type const &r)
{
    parallel::parallel_for(r, [&](range_type const &r) { for (auto const &s:r) { m_data_->erase(s); }});

}
//**************************************************************************************************



template<typename P, typename M, typename ...Policies>
template<typename OutputIterator> OutputIterator
Particle<P, M, Policies...>::copy(id_type s, OutputIterator out_it) const
{
    typename container_type::const_accessor c_accessor;
    if (m_data_->find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename P, typename M, typename ...Policies>
template<typename OutputIterator> OutputIterator
Particle<P, M, Policies...>::copy(range_type const &r, OutputIterator out_it) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy(s, out_it); }
    return out_it;
}

template<typename P, typename M, typename ...Policies>
template<typename OutputIterator> OutputIterator
Particle<P, M, Policies...>::copy(OutputIterator out_it) const
{
    return copy(m_mesh_->template range<iform>(), out_it);
}
//*******************************************************************************


template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::merge(id_type const &s, container_type *buffer)
{
    typename container_type::accessor acc0;
    typename container_type::accessor acc1;

    if (buffer->find(acc1, s))
    {
        m_data_->insert(acc0, s);
        acc0->second.splice(acc0->second.end(), acc1->second);
    }
}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::merge(buffer_type *buffer, container_type *other)
{
    for (auto &item:*buffer)
    {
        typename container_type::accessor acc1;

        other->insert(acc1, item.first);

        acc1->second.splice(acc1->second.end(), item.second);
    }
};

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::merge(range_type const &r, container_type *other)
{
    for (auto const &s:r) { merge(s, other); }
}


template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::merge(container_type *other)
{
    m_mesh_->template for_each_ghost<iform>([&](range_type const &r) { merge(r, other); });

    m_mesh_->template for_each_boundary<iform>([&](range_type const &r) { merge(r, other); });

    m_mesh_->template for_each_center<iform>([&](range_type const &r) { merge(r, other); });
}
//*******************************************************************************

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::sync(container_type const &buffer, parallel::DistributedObject *dist_obj,
                                  bool update_ghost)
{

    data_model::DataType d_type = data_model::DataType::create<value_type>();

    typename mesh_type::index_tuple memory_min, memory_max;
    typename mesh_type::index_tuple local_min, local_max;

    std::tie(memory_min, memory_max) = m_mesh_->memory_index_box();
    std::tie(local_min, local_max) = m_mesh_->local_index_box();


    for (unsigned int tag = 0, tag_e = (1U << (m_mesh_->ndims * 2)); tag < tag_e; ++tag)
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

                auto send_range = m_mesh_->template make_range<iform>(send_min, send_max);

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
                THROW_EXCEPTION_RUNTIME_ERROR("add communication link error", error.what());

            }
        }
    }
}


template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::rehash(id_type const &key, container_type *other)
{
    typename container_type::accessor acc0;

    if (m_data_->find(acc0, key))
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

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::rehash(range_type const &r, container_type *buffer)
{
    for (auto const &s:r) { rehash(s, buffer); }
}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::rehash()
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
    m_mesh_->template for_each_boundary<iform>([&](range_type const &r) { rehash(r, &buffer); });


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
    m_mesh_->template for_each_ghost<iform>([&](range_type const &r) { rehash(r, &buffer); });
    /**
     *
     *  ***************************
     *  *     ............        *
     *  *     .  center  .        *
     *  *     .  <===>   .        *
     *  *     ............        *
     *  ***************************
     */
    m_mesh_->template for_each_center<iform>([&](range_type const &r) { rehash(r, &buffer); });


    //collect moved particle
    m_mesh_->template for_each_center<iform>([&](range_type const &r) { merge(r, &buffer); });

    m_mesh_->template for_each_boundary<iform>([&](range_type const &r) { merge(r, &buffer); });

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

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::push_back(value_type const &p)
{
    typename container_type::accessor acc;
    m_data_->insert(acc, hash(p));
    acc->second.push_back(p);
}

template<typename P, typename M, typename ...Policies>
template<typename InputIterator> void
Particle<P, M, Policies...>::push_back(InputIterator const &b, InputIterator const &e)
{
    // fixme need parallelism
    std::map<id_type, bucket_type> buffer;
    for (auto it = b; it != e; ++it)
    {
        buffer[hash(*it)].push_back(*it);
    }
    this->merge(&buffer, m_data_.get());
}


template<typename P, typename M, typename ...Policies> data_model::DataSet
Particle<P, M, Policies...>::data_set() const
{

    auto r = m_mesh_->template range<iform>();
    size_t count = static_cast<int>(size(r));

    auto data = sp_alloc_memory(count * sizeof(value_type));

    copy(r, reinterpret_cast<value_type *>( data.get()));

    data_model::DataSet ds = data_model::DataSet::create(data_model::DataType::create<value_type>(), data, count);

    return std::move(ds);
}

template<typename P, typename M, typename ...Policies> void
Particle<P, M, Policies...>::data_set(data_model::DataSet const &ds)
{
    size_t count = ds.memory_space.size();

    value_type const *p = reinterpret_cast<value_type const *>(ds.data.get());

    push_back(p, p + count);

//    engine_type::properties.append(ds.properties);

    rehash();
}


template<typename P, typename M, typename ...Policies>
template<typename TGen, typename ...Args> void
Particle<P, M, Policies...>::generator(id_type s, TGen &gen, size_t pic, Args &&...args)
{
    auto g = gen.generator(pic, m_mesh_->volume(s), m_mesh_->box(s),
                           std::forward<Args>(args)...);

    typename container_type::accessor acc;

    m_data_->insert(acc, s);

    std::copy(std::get<0>(g), std::get<1>(g), std::back_inserter(acc->second));
}


template<typename P, typename M, typename ...Policies>
template<typename TGen, typename ...Args> void
Particle<P, M, Policies...>::generator(const range_type &r, TGen &gen, size_t pic, Args &&...args)
{
    for (auto const &s:r) { generator(s, gen, pic, std::forward<Args>(args)...); }
}


template<typename P, typename M, typename ...Policies>
template<typename TGen, typename ...Args> void
Particle<P, M, Policies...>::generator(TGen &gen, size_t pic, Args &&...args)
{
//    m_mesh_->template for_each_ghost<iform>([&](range_type const &r) {
// generator(r, std::forward<Args>(args)...); });

    size_t num_of_particle = m_mesh_->template range<iform>().size() * pic;

    gen.reserve(num_of_particle);

    m_mesh_->template for_each_boundary<iform>(
            [&](range_type const &r)
            {
                generator(r, gen, pic, std::forward<Args>(args)...);
            });

    parallel::DistributedObject dist_obj;

    sync(*m_data_, &dist_obj, false);

    dist_obj.sync();

    m_mesh_->template for_each_center<iform>(
            [&](range_type const &r)
            {
                generator(r, gen, pic, std::forward<Args>(args)...);
            });

    dist_obj.wait();

    for (auto const &item :  dist_obj.recv_buffer)
    {
        value_type const *p = reinterpret_cast<value_type const *>(std::get<1>(item).data.get());
        push_back(p, p + std::get<1>(item).memory_space.size());
    }
}

template<typename P, typename M, typename ...Policies>
template<typename TRange, typename Predicate> void
Particle<P, M, Policies...>::remove_if(TRange const &r, Predicate const &pred)
{
    parallel::parallel_for(
            r,
            [&](TRange const &r)
            {
                for (auto const &s:r)
                {
                    typename container_type::accessor acc;

                    if (m_data_->find(acc, std::get<0>(s)))
                    {
                        acc->second.remove_if([&](value_type const &p) { return pred(s, p); });
                    }


                }
            }

    );
}

template<typename P, typename M, typename ...Policies>
template<typename TConstraint, typename TFun> void
Particle<P, M, Policies...>::accept(TConstraint const &constraint, TFun const &fun)
{
    container_type buffer;
    parallel::parallel_for(
            constraint.range(),
            [&](typename TConstraint::range_type const &r)
            {
                for (auto const &item:r)
                {

                    typename container_type::accessor acc;

                    if (m_data_->find(acc, std::get<0>(item)))
                    {
                        for (auto &p:acc->second) { fun(item, &p); }
                    }

                    rehash(std::get<0>(item), &buffer);
                }
            }

    );
    merge(&buffer);
}


}}//namespace simpla


#endif /* CORE_PARTICLE_PARTICLE_H_ */
