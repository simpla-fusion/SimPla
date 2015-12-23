/**
 * @file ParticleContainer.h
 *
 *  Created on: 2015-3-26
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_CONTAINER_H_
#define CORE_PARTICLE_PARTICLE_CONTAINER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../data_model/DataSet.h"

#include "../gtl/utilities/utilities.h"

#include "../gtl/primitives.h"



/** @ingroup physical_object
*  @addtogroup particle particle
*  @{
*	  @brief  @ref particle  is an abstraction from  physical particle or "phase-space sample".
*	  @details
* ## Summary
*  - @ref particle is used to  describe trajectory in  @ref phase_space_7d  .
*  - @ref particle is used to  describe the behavior of  discrete samples of
*    @ref phase_space_7d function  \f$ f\left(t,x,y,z,v_x,v_y,v_z \right) \f$.
*  - @ref particle is a @ref container;
*  - @ref particle is @ref splittable;
*  - @ref particle is a @ref field
* ### Data Structure
*  -  @ref particle is  `unorder_set<Point_s>`
*
* ## Requirements
*- The following table lists the requirements of a particle type  '''P'''
*	Pseudo-Signature    | Semantics
* -------------------- |----------
* ` struct Point_s `   | data  type of sample point
* ` P( ) `             | Constructor
* ` ~P( ) `            | Destructor
* ` void  next_time_step(dt, args ...) const; `  | push  particles a time interval 'dt'
* ` void  next_time_step(num_of_steps,t0, dt, args ...) const; `  | push  particles from time 't0' to 't1' with time step 'dt'.
* ` flush_buffer( ) `  | flush input buffer to internal data container
*
*- @ref particle meets the requirement of @ref container,
* Pseudo-Signature                 | Semantics
* -------------------------------- |----------
* ` push_back(args ...) `          | Constructor
* ` foreach(TFun const & fun)  `   | Destructor
* ` dataset dump() `               | dump/copy 'data' into a data_set
*
*- @ref particle meets the requirement of @ref physical_object
*   Pseudo-Signature           | Semantics
* ---------------------------- |----------
* ` print(std::ostream & os) ` | print decription of object
* ` update() `                 | update internal data storage and prepare for execute 'next_time_step'
* ` sync()  `                  | sync. internal data with other processes and threads
*
*
* ## Description
* @ref particle   consists of  @ref particle_container and @ref particle_engine .
*   @ref particle_engine  describes the individual behavior of one generator. @ref particle_container
*	  is used to manage these samples.
*
*
* ## Example
*
*  @}
*/
namespace simpla { namespace particle
{

template<typename P, typename M, typename ...Policies>
struct ParticleContainer :
        public M::AttributeEntity,
        public P,
        public Policies ...,
        public std::enable_shared_from_this<ParticleContainer<P, M, Policies...>>
{

public:
    typedef M mesh_type;
    typedef P engine_type;
    typedef typename M::AttributeEntity mesh_entity;

    typedef ParticleContainer<P, M, Policies...> this_type;


    typedef typename this_type::interpolate_policy interpolate_policy;

    typedef typename engine_type::point_type point_type;
    typedef point_type value_type;

    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;
    typedef std::list<value_type> bucket_type;
    typedef parallel::concurrent_hash_map<id_type, bucket_type> container_type;

    container_type m_data_;
    Properties m_properties_;

public:

    static constexpr int iform = VOLUME;

    typedef std::map<id_type, bucket_type> buffer_type;

    static constexpr int ndims = mesh_type::ndims;


public:


    ParticleContainer(M const &m, std::string const &s_name);

    ParticleContainer(this_type const &) = delete;

    ParticleContainer(this_type &&) = delete;

    this_type &operator=(this_type const &other) = delete;

    void swap(this_type const &other) = delete;

    virtual ~ParticleContainer();


    virtual Properties &properties() { return m_properties_; };

    virtual Properties const &properties() const { return m_properties_; };


    virtual void deploy();

    virtual void clear();

    virtual void sync();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;


    data_model::DataSet data_set() const;

    void data_set(data_model::DataSet const &);

    //! @name as container
    //! @{

    /**
     * @require TConstraint = map<id_type,TARGS>
     *          Fun         = function<void(TARGS const &,sample_type*)>
     */
    template<typename TConstraint, typename TFun> void accept(TConstraint const &, TFun const &fun);

    template<typename TRange, typename Predicate> void remove_if(TRange const &r, Predicate const &pred);


    void push_back(value_type const &p);

    template<typename InputIteratorerator>
    void push_back(InputIteratorerator const &b, InputIteratorerator const &e);

    void erase(id_type const &k) { m_data_.erase(k); };

    void erase(range_type const &r);

    void erase_all() { m_data_.clear(); }


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


    //! @name as particle
    //! @{

    constexpr id_type hash(value_type const &p) const
    {
        return mesh_entity::mesh().id(engine_type::project(p), mesh_type::node_id(iform));
    }

    template<typename TGen, typename ...Args> void generator(id_type s, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename TRange, typename ...Args>
    void generator(TRange const &, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generator(TGen &gen, size_t pic, Args &&...args);


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



    void sync(container_type const &buffer, parallel::DistributedObject *dist_obj, bool update_ghost = true);

};//class ParticleContainer

template<typename P, typename M, typename ...Policies>
ParticleContainer<P, M, Policies...>::ParticleContainer(M const &m, std::string const &s_name)
        : mesh_entity(m)
{
    properties()["Name"] = s_name;
}


template<typename P, typename M, typename ...Policies>
ParticleContainer<P, M, Policies...>::~ParticleContainer() { }


template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::clear()
{
    deploy();
    m_data_.clear();
}

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::deploy()
{
    if (properties().click() > this->click())
    {
        engine_type::deploy();
        this->touch();
    }
}

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::sync() { rehash(); };


template<typename P, typename M, typename ...Policies>
std::ostream &ParticleContainer<P, M, Policies...>::print(std::ostream &os, int indent) const
{
    mesh_entity::print(os, indent + 1);
    return os;
}


//*******************************************************************************

template<typename P, typename M, typename ...Policies>
template<typename TField> void
ParticleContainer<P, M, Policies...>::integral(id_type const &s, TField *J) const
{
    static constexpr int f_iform = traits::iform<TField>::value;

    auto x0 = mesh_entity::mesh().point(s);


    id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = mesh_entity::mesh().get_adjacent_cells(iform, s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (m_data_.find(acc1, neighbours[i]))
        {
            for (auto const &p:acc1->second)
            {
                typename ::simpla::traits::field_value_type<TField>::type v;

                engine_type::integral(x0, p, &v);

                (*J)[s] += interpolate_policy::template sample<f_iform>(mesh_entity::mesh(), s, v);
            }
        }
    }
};

template<typename P, typename M, typename ...Policies>
template<typename TField> void
ParticleContainer<P, M, Policies...>::integral(range_type const &r, TField *J) const
{
    // TODO cache J, Base on r
    for (auto const &s:r) { integral(s, J); }
};

template<typename P, typename M, typename ...Policies>
template<typename TField> void
ParticleContainer<P, M, Policies...>::integral(TField *J) const
{

    CMD << "integral particle [" << mesh_entity::name()
    << "] to Field [" << J->attribute()->name() << "<" << J->attribute()->center_type() << ","
    << J->attribute()->extent(0) << ">]" << std::endl;


    static constexpr int f_iform = traits::iform<TField>::value;
    mesh_entity::mesh().template for_each_boundary<f_iform>([&](range_type const &r) { integral(r, J); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    mesh_entity::mesh().template for_each_center<f_iform>([&](range_type const &r) { integral(r, J); });

    dist_obj.wait();


}
//*******************************************************************************

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
ParticleContainer<P, M, Policies...>::push(id_type const &s, Args &&...args)
{
    typename container_type::accessor acc;

    if (m_data_.find(acc, s))
    {
        for (auto &p:acc->second)
        {
            engine_type::push(&p, std::forward<Args>(args)...);
        }
    }


};

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
ParticleContainer<P, M, Policies...>::push(range_type const &r, Args &&...args)
{
    // TODO cache args, Base on s or r
    for (auto const &s:r) { push(s, std::forward<Args>(args)...); }
};

template<typename P, typename M, typename ...Policies>
template<typename ...Args> void
ParticleContainer<P, M, Policies...>::push(Args &&...args)
{


    CMD << "Push particle [" << mesh_entity::name() << "]" << std::endl;

    mesh_entity::mesh().template for_each_ghost<iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    mesh_entity::mesh().template for_each_boundary<iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    mesh_entity::mesh().template for_each_center<iform>(
            [&](range_type const &r) { push(r, std::forward<Args>(args)...); });

    rehash();
}


//**************************************************************************************************

template<typename P, typename M, typename ...Policies> size_t
ParticleContainer<P, M, Policies...>::size(range_type const &r, container_type const &c) const
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
ParticleContainer<P, M, Policies...>::size(range_type const &r) const
{
    return size(r, m_data_);
}

template<typename P, typename M, typename ...Policies> size_t
ParticleContainer<P, M, Policies...>::size() const
{
    return size(mesh_entity::mesh().template range<iform>(), m_data_);
}


//**************************************************************************************************

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::erase(range_type const &r)
{
    parallel::parallel_for(r, [&](range_type const &r) { for (auto const &s:r) { m_data_.erase(s); }});

}
//**************************************************************************************************



template<typename P, typename M, typename ...Policies>
template<typename OutputIterator> OutputIterator
ParticleContainer<P, M, Policies...>::copy(id_type s, OutputIterator out_it) const
{
    typename container_type::const_accessor c_accessor;
    if (m_data_.find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename P, typename M, typename ...Policies>
template<typename OutputIterator> OutputIterator
ParticleContainer<P, M, Policies...>::copy(range_type const &r, OutputIterator out_it) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy(s, out_it); }
    return out_it;
}

template<typename P, typename M, typename ...Policies>
template<typename OutputIterator> OutputIterator
ParticleContainer<P, M, Policies...>::copy(OutputIterator out_it) const
{
    return copy(mesh_entity::mesh().template range<iform>(), out_it);
}
//*******************************************************************************


template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::merge(id_type const &s, container_type *buffer)
{
    typename container_type::accessor acc0;
    typename container_type::accessor acc1;

    if (buffer->find(acc1, s))
    {
        m_data_.insert(acc0, s);
        acc0->second.splice(acc0->second.end(), acc1->second);
    }
}

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::merge(buffer_type *buffer, container_type *other)
{
    for (auto &item:*buffer)
    {
        typename container_type::accessor acc1;

        other->insert(acc1, item.first);

        acc1->second.splice(acc1->second.end(), item.second);
    }
};

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::merge(range_type const &r, container_type *other)
{
    for (auto const &s:r) { merge(s, other); }
}


template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::merge(container_type *other)
{
    mesh_entity::mesh().template for_each_ghost<iform>([&](range_type const &r) { merge(r, other); });

    mesh_entity::mesh().template for_each_boundary<iform>([&](range_type const &r) { merge(r, other); });

    mesh_entity::mesh().template for_each_center<iform>([&](range_type const &r) { merge(r, other); });
}
//*******************************************************************************

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::sync(container_type const &buffer, parallel::DistributedObject *dist_obj,
                                           bool update_ghost)
{

    data_model::DataType d_type = data_model::DataType::create<value_type>();

    typename mesh_type::index_tuple memory_min, memory_max;
    typename mesh_type::index_tuple local_min, local_max;

    memory_min = traits::get<0>(mesh_entity::mesh().memory_index_box());
    memory_max = traits::get<1>(mesh_entity::mesh().memory_index_box());

    local_min = traits::get<0>(mesh_entity::mesh().local_index_box());
    local_max = traits::get<1>(mesh_entity::mesh().local_index_box());


    for (unsigned int tag = 0, tag_e = (1U << (mesh_entity::mesh().ndims * 2)); tag < tag_e; ++tag)
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

                auto send_range = mesh_entity::mesh().template make_range<iform>(send_min, send_max);

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
                RUNTIME_ERROR << "add communication link error" << error.what() << std::endl;

            }
        }
    }
}


template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::rehash(id_type const &key, container_type *other)
{
    typename container_type::accessor acc0;

    if (m_data_.find(acc0, key))
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
ParticleContainer<P, M, Policies...>::rehash(range_type const &r, container_type *buffer)
{
    for (auto const &s:r) { rehash(s, buffer); }
}

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::rehash()
{
    if (properties()["DisableRehash"]) { return; }

    CMD << "Rehash particle [" << properties()["Name"] << "]" << std::endl;

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


    mesh_entity::mesh().template for_each_boundary<iform>([&](range_type const &r) { rehash(r, &buffer); });


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


    mesh_entity::mesh().template for_each_ghost<iform>([&](range_type const &r) { rehash(r, &buffer); });
    /**
     *
     *  ***************************
     *  *     ............        *
     *  *     .  center  .        *
     *  *     .  <===>   .        *
     *  *     ............        *
     *  ***************************
     */

    mesh_entity::mesh().template for_each_center<iform>([&](range_type const &r) { rehash(r, &buffer); });

    //collect moved particle
    mesh_entity::mesh().template for_each_center<iform>([&](range_type const &r) { merge(r, &buffer); });

    mesh_entity::mesh().template for_each_boundary<iform>([&](range_type const &r) { merge(r, &buffer); });

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
ParticleContainer<P, M, Policies...>::push_back(value_type const &p)
{
    typename container_type::accessor acc;

    m_data_.insert(acc, hash(p));

    acc->second.push_back(p);
}

template<typename P, typename M, typename ...Policies>
template<typename InputIterator> void
ParticleContainer<P, M, Policies...>::push_back(InputIterator const &b, InputIterator const &e)
{
    // fixme need parallelism
    std::map<id_type, bucket_type> buffer;

    for (auto it = b; it != e; ++it) { buffer[hash(*it)].push_back(*it); }

    this->merge(&buffer, &m_data_);
}


template<typename P, typename M, typename ...Policies> data_model::DataSet
ParticleContainer<P, M, Policies...>::data_set() const
{
    auto r = mesh_entity::mesh().template range<iform>();

    size_t count = static_cast<int>(size(r));

    auto data = sp_alloc_memory(count * sizeof(value_type));

    copy(r, reinterpret_cast<value_type *>( data.get()));

    data_model::DataSet ds = data_model::DataSet::create(data_model::DataType::create<value_type>(), data, count);

    return std::move(ds);
}

template<typename P, typename M, typename ...Policies> void
ParticleContainer<P, M, Policies...>::data_set(data_model::DataSet const &ds)
{
    UNIMPLEMENTED;
}


template<typename P, typename M, typename ...Policies>
template<typename TGen, typename ...Args> void
ParticleContainer<P, M, Policies...>::generator(id_type s, TGen &gen, size_t pic, Args &&...args)
{
    auto g = gen.generator(pic, mesh_entity::mesh().volume(s), mesh_entity::mesh().box(s),
                           std::forward<Args>(args)...);

    typename container_type::accessor acc;

    m_data_.insert(acc, s);

    std::copy(std::get<0>(g), std::get<1>(g), std::back_inserter(acc->second));
}


template<typename P, typename M, typename ...Policies>
template<typename TGen, typename TRange, typename ...Args> void
ParticleContainer<P, M, Policies...>::generator(const TRange &r, TGen &gen, size_t pic, Args &&...args)
{
    for (auto const &s:r) { generator(s, gen, pic, std::forward<Args>(args)...); }
}


template<typename P, typename M, typename ...Policies>
template<typename TGen, typename ...Args> void
ParticleContainer<P, M, Policies...>::generator(TGen &gen, size_t pic, Args &&...args)
{
//    mesh_entity::mesh().template for_each_ghost<iform>([&](range_type const &r) {
// generator(r, std::forward<Args>(args)...); });

    size_t num_of_particle = mesh_entity::mesh().template range<iform>().size() * pic;

    gen.reserve(num_of_particle);

    mesh_entity::mesh().template for_each_boundary<iform>(
            [&](range_type const &r)
            {
                generator(r, gen, pic, std::forward<Args>(args)...);
            });

    parallel::DistributedObject dist_obj;

    sync(m_data_, &dist_obj, false);

    dist_obj.sync();

    mesh_entity::mesh().template for_each_center<iform>(
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
ParticleContainer<P, M, Policies...>::remove_if(TRange const &r, Predicate const &pred)
{
    parallel::parallel_for(
            r,
            [&](TRange const &r)
            {
                for (auto const &s:r)
                {
                    typename container_type::accessor acc;

                    if (m_data_.find(acc, std::get<0>(s)))
                    {
                        acc->second.remove_if([&](value_type const &p) { return pred(s, p); });
                    }


                }
            }

    );
}

template<typename P, typename M, typename ...Policies>
template<typename TConstraint, typename TFun> void
ParticleContainer<P, M, Policies...>::accept(TConstraint const &constraint, TFun const &fun)
{
    container_type buffer;
    parallel::parallel_for(
            constraint.range(),
            [&](typename TConstraint::range_type const &r)
            {
                for (auto const &item:r)
                {

                    typename container_type::accessor acc;

                    if (m_data_.find(acc, std::get<0>(item)))
                    {
                        for (auto &p:acc->second) { fun(item, &p); }
                    }

                    rehash(std::get<0>(item), &buffer);
                }
            }

    );
    merge(&buffer);
}


}} //namespace simpla { namespace particle







#endif /* CORE_PARTICLE_PARTICLE_CONTAINER_H_ */
