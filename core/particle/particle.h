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

#include "../gtl/design_pattern/singleton_holder.h"
#include "../gtl/utilities/memory_pool.h"

namespace simpla
{
template<typename...> struct Particle;


template<typename P, typename M>
struct Particle<P, M>
        : public P,
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
    using engine_type::function_value;

    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;


    typedef std::list<value_type> bucket_type;

    typedef std::map<id_type, bucket_type> container_type;

    static constexpr int ndims = mesh_type::ndims;

    mesh_type const &m_mesh_;

    container_type m_data_;
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

    //! @ingroup as fiber bundle @{
    // @note this should be lock free on Particle
    template<typename TField>
    void integral(id_type const &s, TField *res) const;

    // @note this should be lock free on Particle
    template<typename TField>
    void integral(range_type const &, TField *res) const;

    // @note this should be lock free on Particle
    template<typename TField>
    void integral(TField *res) const;

    // @note this should be lock free on Particle
    template<typename ...Args>
    void push(id_type const &s, Args &&...args);


    // @note this should be lock free on Particle
    template<typename ...Args>
    void push(range_type const &, Args &&...args);

    // @note this should be lock free on Particle
    template<typename ...Args>
    void push(Args &&...args);


//! @}

//! @ingroup as container bundle @{

    id_type hash(value_type const &p) const { return m_mesh_.id(project(p), mesh_type::node_id(iform)); }

    bucket_type &operator[](id_type const &k) { return m_data_[k]; }

    bucket_type &at(id_type const &k) { return m_data_.at(k); }

    bucket_type const &at(id_type const &k) const { return m_data_.at(k); }


    bucket_type &equal_range(value_type const &p) { return m_data_.at(hash(p)); }

    bucket_type const &equal_range(value_type const &p) const { return m_data_.at(hash(p)); }

    DataSet dataset() const;

    void dataset(DataSet const &);

    void push_back(value_type const &p) { m_data_[hash(p)].push_back(p); }

    template<typename InputIter>
    void push_back(InputIter const &b, InputIter const &e);

    void erase(id_type const &key) { m_data_.erase(key); }

    template<typename TRange>
    void erase(TRange const &r);

    void erase_all() { m_data_.clear(); }

    void clear() { m_data_.clear(); }


    //! @}
    //! @{
    size_t size(id_type const &s, container_type const &buffer) const;

    size_t size(id_type const &s) const;

    template<typename ...Args> size_t size(range_type const &r, Args &&...args) const;

    size_t size() const;

    //! @}
    //! @{


    template<typename OutputIT> OutputIT copy(id_type s, OutputIT out_it, container_type const &buffer) const;

    template<typename OutputIT> OutputIT copy(id_type s, OutputIT out_it) const;

    template<typename OutputIT, typename ...Args>
    OutputIT copy(range_type const &r, OutputIT out_it, Args &&...args) const;

    template<typename OutputIT> OutputIT copy(OutputIT out_it) const;




    //! @}
    //! @{



    void merge(id_type const &s, container_type *buffer);


    template<typename ...Args> void merge(range_type const &r, Args &&...);

    template<typename ...Args> void merge(Args &&...args);

    void merge(this_type *other) { merge(&(other->m_data_)); };

    //! @}
    //! @{

    void rehash(id_type const &s, container_type *buffer);

    template<typename TRange>
    void rehash(TRange const &r, container_type *buffer);

    void rehash();

    //! @}


    template<typename TGen, typename ...Args> void generator(id_type s, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(range_type const &, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(TGen &gen, size_t pic, Args &&...args);

private:
    void sync(container_type const &buffer, parallel::DistributedObject *dist_obj);

};//class Particle
namespace traits
{
template<typename P, typename M>
struct iform<Particle<P, M>> :
        public std::integral_constant<int, Particle<P, M>::iform>
{
};

template<typename P, typename M>
struct value_type<Particle<P, M>>
{
    typedef typename Particle<P, M>::sample_type type;
};


}

//namespace traits
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
    if (m_data_.find(s) == m_data_.end()) { return; }

    static constexpr int MAX_NEIGHBOUR_NUM = 12;
    id_type neighbour[MAX_NEIGHBOUR_NUM];
    auto x0 = m_mesh_.point(s);
    // fixme temporary remove
//        int num = m_mesh_.get_neighbour(s, iform);
//
//        for (int i = 0; i < num; ++i)
//            for (auto const &p:(*this)[neighbour[i]])
//            {
//                (*J)[s] += m_mesh_.RBF(project(p), x0) *
//                           m_mesh_.generator(s, engine_type::integral_v(p));
//            }

};

template<typename P, typename M>
template<typename TField>
void Particle<P, M>::integral(range_type const &r, TField *J) const
{
    // TODO cache J, base on r
    for (auto const &s:r)
    {
        integral(s, J);
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
//*******************************************************************************

template<typename P, typename M>
template<typename ...Args>
void Particle<P, M>::push(id_type const &s, Args &&...args)
{
    if (m_data_.find(s) == m_data_.end()) { return; }

    for (auto &p:m_data_[s])
    {
        engine_type::push(&p, std::forward<Args>(args)...);
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
}


//**************************************************************************************************

template<typename P, typename M>
size_t Particle<P, M>::size(id_type const &s, container_type const &buffer) const
{
    return (buffer.find(s) != buffer.end()) ? buffer.at(s).size() : 0;
}

template<typename P, typename M>
size_t Particle<P, M>::size(id_type const &s) const
{
    return size(s, m_data_);
}

template<typename P, typename M>
template<typename ...Args>
size_t Particle<P, M>::size(range_type const &r, Args &&...args) const
{
    size_t count = 0;
//    parallel::parallel_reduce(r, [&](range_type const &r)
    {
        for (auto const &s:r)
        {
            count += size(s, std::forward<Args>(args)...);
        }
    }
//    );

    return count;
}

template<typename P, typename M>
size_t Particle<P, M>::size() const
{
    size_t count = 0;
    for (auto const &b:m_data_)
    {
        count += b.second.size();
    }

    return count;
}


//**************************************************************************************************

template<typename P, typename M>
template<typename TRange>
void Particle<P, M>::erase(TRange const &r)
{
    for (auto const &s:r)
    {
        erase(s);
    }
}
//**************************************************************************************************

template<typename P, typename M>
template<typename OutputIter>
OutputIter Particle<P, M>::copy(id_type s, OutputIter out_it, container_type const &buffer) const
{
    if (buffer.find(s) != buffer.end())
    {
        out_it = std::copy(buffer.at(s).begin(), buffer.at(s).end(), out_it);
    }

    return out_it;
}

template<typename P, typename M>
template<typename OutputIter>
OutputIter Particle<P, M>::copy(id_type s, OutputIter out_it) const
{
    return copy(s, out_it, m_data_);
}

template<typename P, typename M>
template<typename OutputIter, typename ...Args>
OutputIter Particle<P, M>::copy(range_type const &r, OutputIter out_it, Args &&...args) const
{
    for (auto const &s:r)
    {
        out_it = copy(s, out_it, std::forward<Args>(args)...);
    }
    return out_it;
}

template<typename P, typename M>
template<typename OutputIter>
OutputIter Particle<P, M>::copy(OutputIter out_it) const
{
    for (auto const &item:m_data_) { out_it = copy(item.first, out_it); }
    return out_it;
}
//*******************************************************************************


template<typename P, typename M>
void Particle<P, M>::merge(id_type const &s, container_type *buffer)
{
    if (m_data_.find(s) == m_data_.end() || buffer->find(s) == buffer->end())
    {
        m_data_[s].splice(m_data_[s].end(), (*buffer)[s]);
    }
}

template<typename P, typename M>
template<typename ...Args>
void Particle<P, M>::merge(range_type const &r, Args &&...args)
{
    for (auto const &s:r) { merge(s, std::forward<Args>(args)...); }
}


template<typename P, typename M>
template<typename ...Args>
void Particle<P, M>::merge(Args &&...args)
{
    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { merge(r, std::forward<Args>(args)...); });

    m_mesh_.template for_each_boundary<iform>([&](range_type const &r) { merge(r, std::forward<Args>(args)...); });

    m_mesh_.template for_each_center<iform>([&](range_type const &r) { merge(r, std::forward<Args>(args)...); });
}
//*******************************************************************************



template<typename P, typename M>
void Particle<P, M>::sync(container_type const &buffer, parallel::DistributedObject *dist_obj)
{

    auto d_type = traits::datatype<value_type>::create();

    typename mesh_type::index_tuple memory_min, memory_max;

    std::tie(memory_min, memory_max) = m_mesh_.memory_index_box();


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

                auto send_range = m_mesh_.template make_range<iform>(send_min, send_max);

                size_t send_size = size(send_range, buffer);

                p_send = SingletonHolder<MemoryPool>::instance().raw_alloc(send_size * sizeof(value_type));

                copy(send_range, reinterpret_cast<value_type *>( p_send.get()), buffer);


                send_buffer.push_back(std::make_tuple(send_size, p_send));

                recv_buffer.push_back(std::make_tuple(0, p_recv));

                dist_obj->add_link_send(&coord_offset[0], send_size, d_type, &p_send);

                dist_obj->add_link_recv(&coord_offset[0], 0, d_type, &p_recv);


            }
            catch (std::exception const &error)
            {
                THROW_EXCEPTION_RUNTIME_ERROR("add coommnication link error", error.what());

            }
        }

    }


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
        value_type *p = reinterpret_cast<value_type *>(std::get<1>(item).get());
        push_back(p, p + std::get<0>(item));
    }
}


template<typename P, typename M>
void Particle<P, M>::rehash(id_type const &key, container_type *buffer)
{

    if (m_data_.find(key) == m_data_.end()) { return; }

    auto &src = m_data_[key];

    auto it = src.begin(), ie = src.end();

    while (it != ie)
    {
        auto p = it;

        ++it;

        auto dest = hash(*p);

        if (dest != key) { (*buffer)[dest].splice((*buffer)[dest].end(), src, p); }
    }

    (*buffer)[key].splice((*buffer)[key].end(), src);

}

template<typename P, typename M>
template<typename InputIt>
void Particle<P, M>::insert(InputIt const &b, InputIt const &e)
{
    // fixme need parallize
    for (auto it = b; it != e; ++it)
    {
        insert(*it);
    }
}


template<typename P, typename M>
template<typename TRange>
void Particle<P, M>::rehash(TRange const &r, container_type *buffer)
{
    for (auto const &s:r) { rehash(s, buffer); }
}


template<typename P, typename M>
DataSet Particle<P, M>::dataset() const
{
    DataSet ds;

    size_t count = static_cast<int>(size());
    size_t offset = 0;
    size_t total_count = count;

    std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));

    ds.dataspace = DataSpace(1, &total_count);

    ds.dataspace.select_hyperslab(&offset, nullptr, &count, nullptr);

    ds.memory_space = DataSpace(1, &count);

    ds.data = SingletonHolder<MemoryPool>::instance().raw_alloc(count * sizeof(value_type));

    copy(reinterpret_cast<value_type *>(ds.data.get()));

    ds.properties.append(engine_type::properties);

    return std::move(ds);
}

template<typename P, typename M>
void Particle<P, M>::dataset(DataSet const &ds)
{
    size_t count = ds.memory_space.size();

    value_type const *p = reinterpret_cast<value_type const *>(ds.data.get());

    insert(p, p + count);

    engine_type::properties.append(ds.properties);

    rehash();
}


template<typename P, typename M>
template<typename TGen, typename ...Args>
void Particle<P, M>::generator(id_type s, TGen &gen, size_t pic, Args &&...args)
{
    auto r = gen.generator(pic, m_mesh_.box<iform>(s));
    std::copy(std::get<0>(r), std::get<1>(r), std::back_inserter(m_data_[s]));
}


template<typename P, typename M>
template<typename TGen, typename ...Args>
void Particle<P, M>::generator(const range_type &range, TGen &gen, size_t pic, Args &&...args)
{
    for (auto const &s:range) { generator(s, gen, pic, std::forward<Args>(args)...); }
}


template<typename P, typename M>
template<typename TGen, typename ...Args>
void Particle<P, M>::generator(TGen &gen, size_t pic, Args &&...args)
{
//    m_mesh_.template for_each_ghost<iform>([&](range_type const &r) { generator(r, std::forward<Args>(args)...); });

    size_t num_of_particle = m_mesh_.template make_range<iform>().size() * pic;
    size_t offset = 0;
    std::tie(offset, num_of_particle) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(num_of_particle));

    gen.discard(offset);


    m_mesh_.template for_each_boundary<iform>(
            [&](range_type const &r) { generator(r, gen, pic, std::forward<Args>(args)...); });

    parallel::DistributedObject dist_obj;

    sync(m_data_, &dist_obj);

    dist_obj.sync();
    m_mesh_.template for_each_center<iform>(
            [&](range_type const &r) { generator(r, gen, pic, std::forward<Args>(args)...); });


    dist_obj.wait();
}


}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_H_ */
