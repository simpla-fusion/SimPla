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
#include "../gtl/containers/UnorderedSet.h"



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
* ` void  next_time_step(dt, args ...) const; `  | push  fluid_sp a time interval 'dt'
* ` void  next_time_step(num_of_steps,t0, dt, args ...) const; `  | push  fluid_sp from time 't0' to 't1' with time step 'dt'.
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
        public gtl::UnorderedSet<typename ParticleEngine::sample_type, typename M::id_type>,
        public std::enable_shared_from_this<ParticleContainer<ParticleEngine, M >>
{

public:

    typedef M mesh_type;

    typedef ParticleEngine engine_type;

    typedef typename M::AttributeEntity mesh_attribute_entity;

    typedef ParticleContainer<ParticleEngine, M> this_type;

    typedef gtl::UnorderedSet<typename ParticleEngine::sample_type, typename M::id_type> container_type;

    typedef typename ParticleEngine::sample_type sample_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;


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

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;


    /**
     *  dump particle position x (point_type)
     */
    virtual data_model::DataSet data_set() const { return checkpoint(); }

    /**
     *  dump all data and information to DataSet
     */
    virtual data_model::DataSet dump() const;


    virtual data_model::DataSet checkpoint() const;


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


    void insert(sample_type const &p) { container_type::insert(p, m_hash_); }

    template<typename InputIterator>
    void insert(InputIterator const &b, InputIterator const &e) { container_type::insert(b, e, m_hash_); }

    void rehash();

    //! @}


    //! @name as particle
    //! @{


    template<typename TGen, typename ...Args> void generate(id_type s, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename TRange, typename ...Args>
    void generate(TRange const &, TGen &gen, size_t pic, Args &&...args);

    template<typename TGen, typename ...Args> void generate(TGen &gen, size_t pic, Args &&...args);


    void sync(container_type const &buffer, parallel::DistributedObject *dist_obj, bool update_ghost = true);

    template<typename Gather>
    void gather(Gather const &g, id_type const &s,
                typename std::template result_of<Gather(sample_type const &)>::type *res) const;

    template<typename Gather>
    void gather(Gather const &g, point_type const &,
                typename std::template result_of<Gather(sample_type const &)>::type *res) const;


    template<typename Gather, typename TRange, typename TField>
    void gather(Gather const &g, TRange const &r, TField *res) const;

    template<typename Gather, typename TField>
    void gather(Gather const &g, TField *res) const;

    using container_type::filter;

    using container_type::remove_if;


};//class ParticleContainer

template<typename P, typename M>
ParticleContainer<P, M>::ParticleContainer(mesh_type &m, std::string const &s_name)
        : mesh_attribute_entity(m), engine_type(m)
{
    m_hash_.m_ = &m;

    m_hash_.engine_ = dynamic_cast<engine_type const *>(this);


    if (s_name != "")
    {
        properties()["Name"] = s_name;
        m.enroll(s_name, this->shared_from_this());
    }
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

template<typename P, typename M> void
ParticleContainer<P, M>::sync() { rehash(); };


template<typename P, typename M>
std::ostream &ParticleContainer<P, M>::print(std::ostream &os, int indent) const
{
    mesh_attribute_entity::print(os, indent + 1);
    return os;
}

//*******************************************************************************
template<typename P, typename M> void
ParticleContainer<P, M>::sync(container_type const &buffer, parallel::DistributedObject *dist_obj,
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

                size_t send_size = buffer.size(send_range);


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
            [&](range_type const &r) { container_type::rehash(r, m_hash_, &buffer); });


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


    mesh_attribute_entity::mesh().template for_each_ghost<iform>(
            [&](range_type const &r) { container_type::rehash(r, m_hash_, &buffer); });
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
            [&](range_type const &r) { container_type::rehash(r, m_hash_, &buffer); });

    //collect moved particle
    mesh_attribute_entity::mesh().template for_each_center<iform>(
            [&](range_type const &r) { container_type::merge(r, &buffer); });

    mesh_attribute_entity::mesh().template for_each_boundary<iform>(
            [&](range_type const &r) { container_type::merge(r, &buffer); });

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

//    for (auto const &item :  dist_obj.recv_buffer)
//    {
//        sample_type const *p = reinterpret_cast<sample_type const *>(std::get<1>(item).data.get());
//        container_type::insert(p, p + std::get<1>(item).memory_space.size(), m_hash_);
//    }

}


template<typename P, typename M>
template<typename TGen, typename ...Args> void
ParticleContainer<P, M>::generate(id_type s, TGen &gen, size_t pic, Args &&...args)
{
    auto g = gen.generator(pic, mesh_attribute_entity::mesh().volume(s), mesh_attribute_entity::mesh().cell_box(s),
                           std::forward<Args>(args)...);


    container_type::insert(std::get<0>(g), std::get<1>(g), s);


}


template<typename P, typename M>
template<typename TGen, typename TRange, typename ...Args> void
ParticleContainer<P, M>::generate(const TRange &r, TGen &gen, size_t pic, Args &&...args)
{
    for (auto const &s:r) { generate(s, gen, pic, std::forward<Args>(args)...); }
}


template<typename P, typename M>
template<typename TGen, typename ...Args> void
ParticleContainer<P, M>::generate(TGen &gen, size_t pic, Args &&...args)
{
//    mesh_attribute_entity::mesh().template for_each_ghost<iform>([&](range_type const &r) {
// generate(r, std::forward<Args>(args)...); });

    size_t num_of_particle = mesh_attribute_entity::mesh().template range<iform>().size() * pic;

    gen.reserve(num_of_particle);

    mesh_attribute_entity::mesh().template for_each_boundary<iform>(
            [&](range_type const &r)
            {
                generate(r, gen, pic, std::forward<Args>(args)...);
            });

    parallel::DistributedObject dist_obj;

    sync(this, &dist_obj, false);

    dist_obj.sync();

    mesh_attribute_entity::mesh().template for_each_center<iform>(
            [&](range_type const &r)
            {
                generate(r, gen, pic, std::forward<Args>(args)...);
            });

    dist_obj.wait();

    for (auto const &item :  dist_obj.recv_buffer)
    {
        sample_type const *p = reinterpret_cast<sample_type const *>(std::get<1>(item).data.get());
        push_back(p, p + std::get<1>(item).memory_space.size());
    }
}


template<typename P, typename M> data_model::DataSet
ParticleContainer<P, M>::dump() const
{
    VERBOSE << "Dump particle [" << this->properties()["Name"] << "]" << std::endl;

    auto r0 = mesh_attribute_entity::mesh().template range<iform>();

    size_t count = static_cast<int>(container_type::size(r0));

    data_model::DataSet ds;

    ds.data_type = data_model::DataType::create<sample_type>();

    ds.data = sp_alloc_memory(count * sizeof(sample_type));

    ds.properties = this->properties();

    std::tie(ds.data_space, ds.memory_space) = data_model::DataSpace::create_simple_unordered(count);

    container_type::copy(reinterpret_cast< sample_type *>( ds.data.get()), r0);


    return std::move(ds);
};


namespace _impl
{

HAS_MEMBER(_tag)

template<typename TP>
auto select_tag(data_model::DataSpace &ds, TP const &p)
-> typename std::enable_if<has_member__tag<TP>::value, void>::type
{
    ds.select_point(p._tag);
}

template<typename TP>
auto select_tag(data_model::DataSpace &ds, TP const &p)
-> typename std::enable_if<!has_member__tag<TP>::value, void>::type
{
}
} //namespace _impl


template<typename P, typename M> data_model::DataSet
ParticleContainer<P, M>::checkpoint() const
{
    VERBOSE << "Save checkpoint of particle [" << this->properties()["Name"] << "]" << std::endl;

    auto r0 = mesh_attribute_entity::mesh().template range<iform>();

    size_t count = static_cast<int>(container_type::size(r0));

    data_model::DataSet ds;

    ds.data_type = data_model::DataType::create<point_type>();

    ds.data = sp_alloc_memory(count * sizeof(point_type));

    std::tie(ds.data_space, ds.memory_space) = data_model::DataSpace::create_simple_unordered(count);


    auto out_it = reinterpret_cast< point_type *>( ds.data.get());

    ds.data_space.clear_selected();

    typename container_type::const_accessor c_accessor;

    for (auto const &s:r0)
    {
        if (container_type::find(c_accessor, s))
        {
            for (auto it = c_accessor->second.begin(), ie = c_accessor->second.end(); it != ie; ++it)
            {
                *out_it = this->mesh().map_to_cartesian(engine_type::project(*it));

                _impl::select_tag(ds.data_space, *it);

                ++out_it;
            }
        }
    }


    return std::move(ds);
}

//*******************************************************************************
template<typename P, typename M> template<typename Gather> void
ParticleContainer<P, M>::gather(Gather const &g, id_type const &s,
                                typename std::template result_of<Gather(sample_type const &)>::type *res) const
{
    id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = mesh_attribute_entity::mesh().get_adjacent_cells(iform, s, neighbours);
    auto x0 = mesh_attribute_entity::mesh().point(s);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (container_type::find(acc1, neighbours[i]))
        {
            auto tmp = *res;
            for (auto const &p:acc1->second) { tmp += g(x0, p); }
            *res += tmp;
        }
    }

};

template<typename P, typename M> template<typename Gather> void
ParticleContainer<P, M>::gather(Gather const &g, point_type const &x0,
                                typename std::template result_of<Gather(sample_type const &)>::type *res) const
{
    id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    id_type s = mesh_attribute_entity::mesh().id(x0);

    int num = mesh_attribute_entity::mesh().get_adjacent_cells(iform, s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (container_type::find(acc1, neighbours[i]))
        {
            auto tmp = *res;
            for (auto const &p:acc1->second) { tmp += g(x0, p); }
            *res += tmp;
        }
    }

};


template<typename P, typename M> template<typename Gather, typename TRange, typename TField> void
ParticleContainer<P, M>::gather(Gather const &g, TRange const &r0, TField *J) const
{
    parallel::parallel_for(r0, [&](TRange const &r)
    {
        for (auto const &s:r)
        {
            (*J)[s] += J->mesh().template sample<iform>(gather(g, s));
        }
    });
};

template<typename P, typename M> template<typename Gather, typename TField> void
ParticleContainer<P, M>::gather(Gather const &g, TField *J) const
{

    CMD << "integral particle [" << mesh_attribute_entity::name()
    << "] to Field [" << J->attribute()->name() << "<" << J->attribute()->center_type() << ","
    << J->attribute()->extent(0) << ">]" << std::endl;

    typedef typename mesh_type::range_type range_t;

    static constexpr int f_iform = traits::iform<TField>::value;

    J->mesh().template for_each_boundary<f_iform>([&](range_t const &r) { gather(g, r, J); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    J->mesh().template for_each_center<f_iform>([&](range_t const &r) { gather(g, r, J); });

    dist_obj.wait();


}


}} //namespace simpla { namespace particle







#endif /* CORE_PARTICLE_PARTICLE_CONTAINER_H_ */
