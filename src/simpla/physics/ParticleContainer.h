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


#include "../toolbox/utilities/utilities.h"
#include "../sp_def.h"
#include "../toolbox/DataSet.h"


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
* ` struct Point_s `   | m_data  type of sample point
* ` P( ) `             | Constructor
* ` ~P( ) `            | Destructor
* ` void  next_time_step(dt, args ...) const; `  | push  m_fluid_sp_ a time interval 'dt'
* ` void  next_time_step(num_of_steps,t0, dt, args ...) const; `  | push  m_fluid_sp_ from time 't0' to 't1' with time step 'dt'.
* ` flush_buffer( ) `  | flush input m_buffer to internal m_data container
*
*- @ref particle meets the requirement of @ref container,
* Pseudo-Signature                 | Semantics
* -------------------------------- |----------
* ` push_back(args ...) `          | Constructor
* ` foreach(TFun const & fun)  `   | Destructor
* ` dataset dump() `               | dump/copy 'm_data' into a dataset
*
*- @ref particle meets the requirement of @ref physical_object
*   Pseudo-Signature           | Semantics
* ---------------------------- |----------
* ` print(std::ostream & os) ` | print decription of object
* ` update() `                 | sync internal m_data storage and prepare for execute 'next_time_step'
* ` sync()  `                  | sync. internal m_data with other processes and threads
*
*
* ## Description
* @ref particle   consists of  @ref particle_container and @ref particle_engine .
*   @ref particle_engine  describes the individual behavior of one generate. @ref particle_container
*	  is used to manage these samples.
*
*
* ## Example
*
*  @}
*/
namespace simpla { namespace particle
{
template<typename ...> struct ParticleContainer;

template<typename ParticleEngine, typename M>
struct ParticleContainer<ParticleEngine, M> :
        public M::AttributeEntity,
        public ParticleEngine,
        public parallel::concurrent_hash_map<typename M::id_type, std::list<typename ParticleEngine::sample_type>>,
        public std::enable_shared_from_this<ParticleContainer<ParticleEngine, M >>
{

public:

    typedef M mesh_type;

    typedef ParticleEngine engine_type;

    typedef typename M::AttributeEntity mesh_attribute_entity;

    typedef ParticleContainer<ParticleEngine, M> this_type;

    typedef typename ParticleEngine::sample_type sample_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;

    typedef sample_type value_type;

    typedef std::list<value_type> bucket_type;

    typedef typename parallel::concurrent_hash_map<id_type, bucket_type> container_type;

    typedef container_type buffer_type;

public:

    HAS_PROPERTIES;

    static constexpr int iform = VOLUME;

    static constexpr int ndims = mesh_type::ndims;

public:


    ParticleContainer(mesh_type &m, std::string const &s_name = "");

    virtual ~ParticleContainer();

    ParticleContainer(this_type const &) = delete;

    ParticleContainer(this_type &&) = delete;

    this_type &operator=(this_type const &other) = delete;

    void swap(this_type const &other) = delete;

    virtual void deploy();

    virtual void clear();

    virtual void sync();

    void sync_(container_type const &buffer, parallel::DistributedObject *dist_obj, bool update_ghost = true);

public:
    virtual std::ostream &print(std::ostream &os, int indent = 1) const;


    /**
     *  dump particle position x (point_type)
     */
    virtual data_model::DataSet data_set() const;

private:
    struct Hash
    {
        mesh_type const *m_;
        engine_type const *engine_;


        id_type operator()(sample_type const &p) const
        {
            return m_->id(engine_->project(p), mesh_type::node_id(iform));
        }
    };

    Hash m_hash_;

public:



    //! @}

    template<typename TFun>
    void foreach_bucket(TFun const &fun, id_type const &s)
    {
        typename container_type::accessor acc;

        if (container_type::find(acc, s)) { fun(acc->second, s); }
    }

    template<typename TFun, typename TRange>
    void foreach_bucket(TFun const &fun, TRange const &r0)
    {
        parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { foreach_bucket(fun, s); }});

    }





    //! @name as container
    //! @{


    void insert(value_type const &p, id_type const &hint);

    void insert(value_type const &p) { insert(p, m_hash_(p)); }

    template<typename InputIterator>
    void insert(InputIterator const &b, InputIterator const &e, id_type const &hint);

    template<typename InputIterator>
    void insert(InputIterator const &, InputIterator const &);

    template<typename Predicate> void remove_if(Predicate const &pred, id_type const &s);

    template<typename Predicate, typename TRange> void remove_if(Predicate const &pred, TRange const &r);

    template<typename Predicate> void remove_if(Predicate const &pred);


    template<typename TFun> void filter(TFun const &fun, id_type const &);

    template<typename TFun> void filter(TFun const &fun, id_type const &) const;

    template<typename TFun> void filter(TFun const &fun, typename container_type::value_type &);

    template<typename TFun> void filter(TFun const &fun, typename container_type::value_type const &) const;

    template<typename TFun, typename TRange> void filter(TFun const &fun, TRange const &);

    template<typename TFun, typename TRange> void filter(TFun const &fun, TRange const &) const;

    template<typename TFun> void filter(TFun const &fun);

    template<typename TFun> void filter(TFun const &fun) const;


    void erase(id_type const &s) { container_type::erase(s); }

    void erase(typename container_type::range_type const &r);

    template<typename TRange> void erase(TRange const &r);

    void erase_all() { container_type::clear(); }


private:

    static size_t count_(container_type const &d, id_type const &s);

    static size_t count_(container_type const &d, typename container_type::value_type const &item);

    template<typename TRange>
    static size_t count_(container_type const &d, TRange const &r);

    static size_t count_(container_type const &d);

public:


    template<typename ...Others> size_t count(Others &&...others) const
    {
        return count_(dynamic_cast<container_type const &>(*this), std::forward<Others>(others)...);
    };

    template<typename OutputIT> OutputIT copy_out(OutputIT out_it, id_type const &s) const;

    template<typename OutputIT, typename TRange> OutputIT copy_out(OutputIT out_it, TRange const &r) const;

    template<typename OutputIT> OutputIT copy_out(OutputIT out_it) const;


    void merge(buffer_type *other, id_type s);

//    void merge(buffer_type *other, typename container_type::value_type &);

    template<typename TRange> void merge(buffer_type *other, TRange const &r);

    void merge(buffer_type *other);


    void rehash(id_type const &key, buffer_type *out_buffer);

    template<typename TRange> void rehash(TRange const &r, buffer_type *out_buffer);

    void rehash(buffer_type *out_buffer);

    void rehash();

    //! @}



};//class ParticleContainer

template<typename P, typename M>
ParticleContainer<P, M>::ParticleContainer(mesh_type &m, std::string const &s_name)
        : mesh_attribute_entity(m), engine_type(m)
{
    m_hash_.m_ = &m;

    m_hash_.engine_ = dynamic_cast<engine_type const *>(this);

    if (s_name != "") { properties()["Name"] = s_name; }
}


template<typename P, typename M>
ParticleContainer<P, M>::~ParticleContainer() { }


template<typename P, typename M> void
ParticleContainer<P, M>::clear()
{
    deploy();
    container_type::clear();
}

template<typename P, typename M> void
ParticleContainer<P, M>::deploy()
{
    if (properties().click() > this->click())
    {
        engine_type::deploy();
        this->touch();
    }
}

template<typename P, typename M>
std::ostream &ParticleContainer<P, M>::print(std::ostream &os, int indent) const
{
    mesh_attribute_entity::print(os, indent + 1);
    os << std::setw(indent + 1) << " " << ", num = " << count() << std::endl;
    return os;
}

template<typename P, typename M> void
ParticleContainer<P, M>::sync() { rehash(); };


//*******************************************************************************
template<typename P, typename M> void
ParticleContainer<P, M>::sync_(container_type const &buffer, parallel::DistributedObject *dist_obj,
                               bool update_ghost)
{

    data_model::DataType d_type = data_model::DataType::create<sample_type>();

    typename mesh_type::index_tuple memory_min, memory_max;
    typename mesh_type::index_tuple local_min, local_max;

    memory_min = traits::get<0>(mesh_attribute_entity::mesh().memory_index_box());
    memory_max = traits::get<1>(mesh_attribute_entity::mesh().memory_index_box());

    local_min = traits::get<0>(mesh_attribute_entity::mesh().local_index_box());
    local_max = traits::get<1>(mesh_attribute_entity::mesh().local_index_box());


    for (unsigned int tag = 0, tag_e = (1U << (mesh_attribute_entity::mesh().ndims * 2)); tag < tag_e; ++tag)
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

                auto send_range = mesh_attribute_entity::mesh().template make_range<iform>(send_min, send_max);

                size_t send_size = count_(buffer, send_range);


                p_send = sp_alloc_memory(send_size * sizeof(sample_type));

                auto p = reinterpret_cast<sample_type *>( p_send.get());


                for (auto const &s:send_range)
                {
                    typename container_type::accessor acc;

                    if (buffer.find(acc, s))
                    {
                        p = std::copy(acc->second.begin(), acc->second.end(), p);
                    }

                }


                dist_obj->add_link_send(coord_offset, d_type, p_send, 1, &send_size);

                dist_obj->add_link_recv(coord_offset, d_type);


            }
            catch (std::exception const &error)
            {
                RUNTIME_ERROR << "add communication link error" << error.what() << std::endl;

            }
        }
    }
}


template<typename P, typename M> void
ParticleContainer<P, M>::rehash()
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


    mesh_attribute_entity::mesh().template for_each_boundary<iform>(
            [&](range_type const &r) { rehash(r, &buffer); });


    //**************************************************************************************
    // Sync ghost area in m_buffer


    parallel::DistributedObject dist_obj;

    sync_(buffer, &dist_obj);
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


    mesh_attribute_entity::mesh().template for_each_ghost<iform>(
            [&](range_type const &r) { rehash(r, &buffer); });
    /**
     *
     *  ***************************
     *  *     ............        *
     *  *     .  center  .        *
     *  *     .  <===>   .        *
     *  *     ............        *
     *  ***************************
     */

    mesh_attribute_entity::mesh().template for_each_center<iform>(
            [&](range_type const &r) { rehash(r, &buffer); });

    //collect moved particle
    mesh_attribute_entity::mesh().template for_each_center<iform>(
            [&](range_type const &r) { merge(&buffer, r); });

    mesh_attribute_entity::mesh().template for_each_boundary<iform>(
            [&](range_type const &r) { merge(&buffer, r); });

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
        sample_type const *p = reinterpret_cast<sample_type const *>(std::get<1>(item).data.get());
        insert(p, p + std::get<1>(item).memory_space.size());
    }
    this->merge(&buffer);

}

template<typename P, typename M> data_model::DataSet
ParticleContainer<P, M>::data_set() const
{
    CMD << "Dump particle [" << this->properties()["Name"] << "] to m_data setValue." << std::endl;


    auto r0 = mesh_attribute_entity::mesh().template range<iform>();

    size_t num = count(r0);

    data_model::DataSet ds;

    ds.data_type = data_model::DataType::create<sample_type>();

    ds.data = sp_alloc_memory(num * sizeof(sample_type));

    ds.properties = this->properties();

    std::tie(ds.data_space, ds.memory_space) = data_model::DataSpace::create_simple_unordered(num);

    copy_out(reinterpret_cast< sample_type *>( ds.data.get()), r0);

    return std::move(ds);
};

template<typename V, typename K> size_t
ParticleContainer<V, K>::count_(container_type const &d, typename container_type::value_type const &item)
{
    return item.second.size();
};

template<typename V, typename K> size_t
ParticleContainer<V, K>::count_(container_type const &d, id_type const &s)
{
    typename container_type::const_accessor acc;

    size_t res = 0;

    if (d.find(acc, s)) { res = acc->second.size(); }


    return res;
}


template<typename V, typename K>
template<typename TRange> size_t
ParticleContainer<V, K>::count_(container_type const &d, TRange const &r0)
{
    return parallel::parallel_reduce(
            r0, 0U,
            [&](TRange const &r, size_t init) -> size_t
            {
                for (auto const &s:r) { init += count_(d, s); }
                return init;
            },
            [](size_t x, size_t y) -> size_t { return x + y; }
    );
}

template<typename V, typename K> size_t
ParticleContainer<V, K>::count_(container_type const &d)
{
    return count_(d, d.range());
//    return parallel::parallel_reduce(
//            d.entity_id_range(), 0U,
//            [&](typename container_type::const_range_type const &r, size_t set_direction) -> size_t
//            {
//                for (auto const &item : r) { set_direction += item.second.size(); }
//
//                return set_direction;
//            },
//            [](size_t x, size_t y) -> size_t
//            {
//                return x + y;
//            }
//    );
}


//**************************************************************************************************


template<typename V, typename K> void
ParticleContainer<V, K>::erase(typename container_type::range_type const &r)
{
    UNIMPLEMENTED;

};


template<typename V, typename K> template<typename TRange> void
ParticleContainer<V, K>::erase(TRange const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { container_type::erase(s); }});
}
//**************************************************************************************************



template<typename V, typename K>
template<typename OutputIterator> OutputIterator
ParticleContainer<V, K>::copy_out(OutputIterator out_it, id_type const &s) const
{
    typename container_type::const_accessor c_accessor;
    if (container_type::find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename V, typename K>
template<typename OutputIT, typename TRange> OutputIT
ParticleContainer<V, K>::copy_out(OutputIT out_it, TRange const &r) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy_out(out_it, s); }
    return out_it;
}

template<typename V, typename K>
template<typename OutputIterator> OutputIterator
ParticleContainer<V, K>::copy_out(OutputIterator out_it) const
{
    return copy_out(out_it, this->range());
}
//*******************************************************************************

template<typename V, typename K> void
ParticleContainer<V, K>::merge(buffer_type *other, id_type s)
{
    typename buffer_type::accessor acc0;
    if (other->find(acc0, s))
    {
        typename container_type::accessor acc1;
        container_type::insert(acc1, s);
        acc1->second.splice(acc1->second.end(), acc0->second);
    }
};


template<typename V, typename K> void
ParticleContainer<V, K>::merge(buffer_type *buffer)
{
    for (auto &item:*buffer)
    {
        typename container_type::accessor acc1;
        container_type::insert(acc1, item.first);
        acc1->second.splice(acc1->second.end(), item.second);
    }

//    merge(m_buffer, m_buffer->entity_id_range());
}

template<typename V, typename K> template<typename TRange> void
ParticleContainer<V, K>::merge(buffer_type *other, TRange const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { merge(other, s); }});

}

//*******************************************************************************


template<typename V, typename K> void
ParticleContainer<V, K>::rehash(id_type const &key, buffer_type *out_buffer)
{
    ASSERT(out_buffer != nullptr);

    typename buffer_type::accessor acc0;

    if (container_type::find(acc0, key))
    {

        auto &src = acc0->second;

        auto it = src.begin(), ie = src.end();

        while (it != ie)
        {
            auto p = it;

            ++it;
            auto s = m_hash_(*p);
            if (s != key)
            {

                typename buffer_type::accessor acc1;

                out_buffer->insert(acc1, s);

                acc1->second.splice(acc1->second.end(), src, p);
            }
        }


    }
    acc0.release();


}

template<typename V, typename K> template<typename TRange> void
ParticleContainer<V, K>::rehash(TRange const &r0, buffer_type *out_buffer)
{
    ASSERT(out_buffer != nullptr);

    parallel::parallel_for(
            r0,
            [&](TRange const &r) { for (auto const &s:r) { rehash(s, out_buffer); }}
    );

}

template<typename V, typename K> void
ParticleContainer<V, K>::rehash(buffer_type *out_buffer)
{
    if (out_buffer == nullptr)
    {
        buffer_type tmp;

        rehash(container_type::range(), &tmp);

        this->merge(&tmp);
    }
    else
    {

        parallel::parallel_for(
                container_type::range(),
                [&](typename container_type::const_range_type &r)
                {
                    for (auto &b:  container_type::range())
                    {
                        for (auto const &p:b.second) { fun(p); }
                    }
                });
        rehash(container_type::range(), out_buffer);
    }
}

//**************************************************************************************************
template<typename V, typename K> void
ParticleContainer<V, K>::insert(value_type const &v, id_type const &s)
{
    typename container_type::accessor acc;

    container_type::insert(acc, s);

    acc->second.push_back(v);
}

template<typename V, typename K> template<typename InputIterator> void
ParticleContainer<V, K>::insert(InputIterator const &b, InputIterator const &e, id_type const &hint)
{
    ASSERT(hint == m_hash_(*b));

    typename container_type::accessor acc;

    container_type::insert(acc, hint);

    acc->second.insert(acc->second.end(), b, e);

}

template<typename V, typename K> template<typename InputIterator> void
ParticleContainer<V, K>::insert(InputIterator const &b, InputIterator const &e)
{
    for (auto it = b; it != e; ++it) { insert(*it); }
}


//*******************************************************************************

template<typename V, typename K> template<typename Predicate> void
ParticleContainer<V, K>::remove_if(Predicate const &pred, id_type const &s)
{
    typename container_type::accessor acc;

    if (container_type::find(acc, std::get<0>(s)))
    {
        acc->second.remove_if([&](value_type const &p) { return pred(p, s); });
    }
}


template<typename V, typename K> template<typename Predicate, typename TRange> void
ParticleContainer<V, K>::remove_if(Predicate const &pred, TRange const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { remove_if(pred, s); }});
}

template<typename V, typename K> template<typename Predicate> void
ParticleContainer<V, K>::remove_if(Predicate const &pred) { remove_if(pred, container_type::range()); }


template<typename V, typename K> template<typename TFun> void
ParticleContainer<V, K>::filter(TFun const &fun, id_type const &s)
{
    typename container_type::accessor acc;

    if (container_type::find(acc, s)) { for (auto &p:acc->second) { fun(&p); }}

};

template<typename V, typename K> template<typename TFun> void
ParticleContainer<V, K>::filter(TFun const &fun, id_type const &s) const
{
    typename container_type::const_accessor acc;

    if (container_type::find(acc, s)) { for (auto const &p:acc->second) { fun(p); }}
};

template<typename V, typename K> template<typename TFun> void
ParticleContainer<V, K>::filter(TFun const &fun, typename container_type::value_type &item)
{
    for (auto &p:item.second) { fun(&p); }
};

template<typename V, typename K> template<typename TFun> void
ParticleContainer<V, K>::filter(TFun const &fun, typename container_type::value_type const &item) const
{
    for (auto const &p:item.second) { fun(p); }
};


template<typename V, typename K>
template<typename TFun, typename TRange> void
ParticleContainer<V, K>::filter(TFun const &fun, TRange const &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto &s:r) { filter(fun, s); }});
}

template<typename V, typename K>
template<typename TFun, typename TRange> void
ParticleContainer<V, K>::filter(TFun const &fun, TRange const &r0) const
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { filter(fun, s); }});
}

template<typename V, typename K> template<typename TFun> void
ParticleContainer<V, K>::filter(TFun const &fun) { filter(fun, container_type::range()); };

template<typename V, typename K> template<typename TFun> void
ParticleContainer<V, K>::filter(TFun const &fun) const { filter(fun, container_type::range()); };

}} //namespace simpla { namespace particle







#endif /* CORE_PARTICLE_PARTICLE_CONTAINER_H_ */
