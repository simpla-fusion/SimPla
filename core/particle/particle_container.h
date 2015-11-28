/**
 * @file particle_container.h
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

#include "../dataset/dataset.h"

#include "../gtl/utilities/utilities.h"

#include "../gtl/primitives.h"

#include "../io/data_stream.h"


/** @ingroup physical_object
*  @addtogroup particle Particle
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
*- The following table lists the requirements of a Particle type  '''P'''
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
* ` dataset dump() `               | dump/copy 'data' into a DataSet
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
namespace simpla
{


/**
 *   `Particle<M,P>` represents a fiber bundle \f$ \pi:P\to M\f$
 *
 */
template<typename ...> struct ParticleContainer;


template<typename P, typename Hasher>
struct ParticleContainer<P, Hasher>
{

private:

    typedef Hasher hasher_type;
    typedef P value_type;

    typedef ParticleContainer<value_type, hasher_type> this_type;


    typedef traits::result_of_t<hasher_type(value_type)> key_type;

    typedef std::list<value_type> bucket_type;

public:

    typedef std::map<key_type, bucket_type> container_type;

private:

    container_type m_data_;

    hasher_type const &m_hasher_;

public:


    ParticleContainer(hasher_type const &m);

    ParticleContainer(this_type const &other);

    ~ParticleContainer();


    DataSet dataset() const;

    void dataset(DataSet const &);

    void insert(value_type const &p) { m_data_[m_hasher_(p)].push_back(p); }

    template<typename IT>
    void insert(IT const &b, IT const &e) { for (auto it = b; it != e; ++it) { insert(*it); }}

    void erase(key_type const &key) { m_data_.erase(key); }

    template<typename TRange>
    void erase(TRange const &r) { for (auto const &s:r) { erase(s); }}

    void clear() { m_data_.clear(); }

    size_t size() const
    {
        size_t count = 0;
        for (auto const &b:m_data_)
        {
            count += b.second.size();
        }

        return count;
    }

    template<typename TRange>
    size_t size(TRange const &r) const
    {
        size_t count = 0;
        for (auto const &s:r)
        {
            if (m_data_.find(s) != m_data_.end())
                count += m_data_[s].size();
        }

        return count;
    }

    bucket_type &operator[](key_type const &k) { return m_data_[k]; }

    bucket_type &at(key_type const &k) { return m_data_.at(k); }

    bucket_type const &at(key_type const &k) const { return m_data_.at(k); }

    template<typename OutputIT>
    OutputIT copy(key_type const &s, OutputIT out_it) const
    {
        if (m_data_.find(s) != m_data_.end())
        {
            out_it = std::copy(m_data_[s].begin(), m_data_[s].end(), out_it);
        }

        return out_it;
    }

    template<typename TRange, typename OutputIT>
    OutputIT copy(TRange const &r, OutputIT out_it) const
    {
        for (auto const &s:r)
        {
            out_it = copy(s, out_it);
        }
        return out_it;

    }

    template<typename OutputIT>
    OutputIT copy(OutputIT out_it) const
    {
        for (auto const &item:m_data_)
        {
            out_it = std::copy(item.second.begin(), item.second.end(), out_it);
        }

        return out_it;
    }


    template<typename TRange>
    void merge(TRange const &r, container_type *buffer);

    void merge(container_type *buffer);

    void merge(this_type *other) { merge(&(other->m_data_)); };


    void rehash(key_type const &s, container_type *buffer);

    template<typename TRange>
    void rehash(TRange const &r, container_type *buffer);

    void rehash()
    {
        container_type buffer;

        for (auto &item:m_data_) { rehash(item.second, &buffer); }

        merge(&buffer);

    }

//! @}

};//class Particle



template<typename P, typename Hasher>
ParticleContainer<P, Hasher>::ParticleContainer(hasher_type const &m) :
        m_hasher_(m)
{

}

template<typename P, typename Hasher>
ParticleContainer<P, Hasher>::ParticleContainer(this_type const &other) :
        m_hasher_(other.m_hasher_), m_data_(other.m_data_)
{
}

template<typename P, typename Hasher>
ParticleContainer<P, Hasher>::~ParticleContainer()
{
}


template<typename P, typename Hasher> void
ParticleContainer<P, Hasher>::rehash(key_type const &key, container_type *buffer)
{

    if (m_data_.find(key) == m_data_.end()) { return; }

    auto &src = m_data_[key];

    auto it = src.begin(), ie = src.end();

    while (it != ie)
    {
        auto p = it;

        ++it;

        auto dest = m_hasher_(*p);

        if (dest != key) { (*buffer)[dest].splice((*buffer)[dest].end(), src, p); }
    }

    (*buffer)[key].splice((*buffer)[key].end(), src);

}

template<typename P, typename Hasher>
template<typename TRange> void
ParticleContainer<P, Hasher>::rehash(TRange const &r, container_type *buffer)
{
    for (auto const &s:r) { rehash(s, buffer); }
}

template<typename P, typename Hasher>
template<typename TRange> void
ParticleContainer<P, Hasher>::merge(TRange const &r, container_type *buffer)
{
    for (auto const &s:r)
    {
        if (buffer->find(s) != buffer->end())
        {
            auto &src = (*buffer)[s];

            auto &dest = m_data_[s];

            dest.splice(dest.end(), src);
        }
    }
};


template<typename P, typename Hasher> void
ParticleContainer<P, Hasher>::merge(container_type *buffer)
{
    for (auto item: (*buffer))
    {
        auto &dest = m_data_[item.first];

        dest.splice(dest.end(), item.second);
    }
};


template<typename P, typename Hasher>
DataSet ParticleContainer<P, Hasher>::dataset() const
{
    DataSet ds;

    size_t count = static_cast<int>(size());
    size_t offset = 0;
    size_t total_count = count;

    std::tie(offset, total_count) = parallel::sync_global_location(GLOBAL_COMM, static_cast<int>(count));

    ds.dataspace = DataSpace(1, &total_count);

    ds.dataspace.select_hyperslab(&offset, nullptr, &count, nullptr);

    ds.memory_space = DataSpace(1, &count);

    ds.data = sp_alloc_memory(count * sizeof(value_type));

    copy(reinterpret_cast<value_type *>(ds.data.get()));

    return std::move(ds);
}

template<typename P, typename Hasher> void
ParticleContainer<P, Hasher>::dataset(DataSet const &ds)
{
    size_t count = ds.memory_space.size();

    value_type const *p = reinterpret_cast<value_type *>(ds.data.get());

    insert(p, p + count);

    rehash();
}


}  // namespace simpla

#endif /* CORE_PARTICLE_PARTICLE_CONTAINER_H_ */
