//
// Created by salmon on 16-6-7.
//

#ifndef SIMPLA_PARTICLELITE_H
#define SIMPLA_PARTICLELITE_H

#include <vector>
#include <list>
#include <map>
#include "../gtl/integer_sequence.h"
#include "../gtl/type_traits.h"
#include "../parallel/Parallel.h"
#include "../mesh/MeshAttribute.h"
#include "../sp/SmallObjPool.h"
#include "ParticlePage.h"

namespace simpla { namespace particle
{

template<typename ...> struct Particle;
typedef typename simpla::tags::VERSION<0, 0, 1> V001;


template<typename P, typename M>
struct Particle<P, M, V001>
        : public P,
          public Field<sp::spPage *, M, index_const<mesh::VOLUME> >,
          public std::enable_shared_from_this<Particle<P, M, V001>>
{
private:

    typedef Particle<P, M, V001> this_type;
    typedef mesh::MeshAttribute::View View;
    typedef mesh::MeshAttribute::View base_type;
public:

    typedef M mesh_type;
    typedef P engine_type;
    typedef Field<sp::spPage *, M, index_const<mesh::VOLUME> > field_type;
    typedef typename P::point_s value_type;
    typedef typename mesh::MeshEntityId id_type;
    typedef typename mesh::MeshEntityRange range_type;


private:
    std::shared_ptr<struct sp::spPagePool> m_pool_;

public:
    virtual Properties const &properties() const
    {
        assert(m_properties_ != nullptr);
        return *m_properties_;
    };


    virtual Properties &properties()
    {
        if (m_properties_ == nullptr) { m_properties_ = std::make_shared<Properties>(); }
        return *m_properties_;
    };

private:
    std::shared_ptr<Properties> m_properties_;
public:


    Particle(mesh_type const *m = nullptr)
            : field_type(m), m_properties_(nullptr), m_pool_(nullptr)
    {
    }

    Particle(mesh::MeshBase const *m)
            : field_type(m), m_properties_(nullptr), m_pool_(nullptr)
    {
        assert(m->template is_a<mesh_type>());
    }

    Particle(std::shared_ptr<base_type> other)
            : field_type(other), m_pool_(nullptr)
    {
        deploy();
    }

    //factory construct
    template<typename TFactory, typename ... Args, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
    Particle(TFactory &factory, Args &&...args)
            : field_type(factory, std::forward<Args>(args)...), m_properties_(nullptr), m_pool_(nullptr)
    {
        deploy();
    }


    //copy construct
    Particle(this_type const &other)
            : engine_type(other), field_type(other), m_properties_(other.m_properties_), m_pool_(other.m_pool_)
    {
    }


    // move construct
    Particle(this_type &&other)
            : engine_type(other), field_type(other), m_properties_(other.m_properties_), m_pool_(other.m_pool_)
    {
    }

    virtual ~Particle() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    virtual void swap(View &other)
    {
        assert(other.template is_a<this_type>());

        swap(dynamic_cast<this_type &>(other));
    }

    void swap(this_type &other)
    {
        field_type::swap(other);

        std::swap(m_properties_, other.m_properties_);
        std::swap(m_pool_, other.m_pool_);

    }

    using field_type::entity_id_range;
    using field_type::entity_type;

    std::ostream &print(std::ostream &os, int indent) const;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); };

    virtual std::string get_class_name() const { return class_name(); };

    static std::string class_name() { return "Particle<" + traits::type_id<P, M>::name() + ">"; };


    virtual bool deploy();


    virtual data_model::DataSet dataset(range_type const &) const;


    virtual void dataset(data_model::DataSet const &);

    virtual void dataset(mesh::MeshEntityRange const &, data_model::DataSet const &);


    virtual size_t size() const { return count(entity_id_range()); }

    virtual void clear();

    //**************************************************************************************************
    // as particle
private:
    HAS_MEMBER_FUNCTION(push);

    HAS_MEMBER_FUNCTION(gather);
public:


    template<typename TRes, typename ...Args,
            typename std::enable_if<has_member_function_gather<TRes *, sp::spPage *, Args...>::value>::type * = nullptr>
    void gather(TRes *res, mesh::point_type const &x0, Args &&...args) const
    {
        *res = 0;

        mesh::MeshEntityId s = std::get<0>(this->m_mesh_->point_global_to_local(x0));

        mesh::MeshEntityId neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

        int num = this->m_mesh_->get_adjacent_entities(this->entity_type(), s, neighbours);

        for (int i = 0; i < num; ++i)
        {
            engine_type::gather(res, this->get(neighbours[i]), x0, std::forward<Args>(args) ...);
        }
    }


    template<typename TRes, typename ...Args,
            typename std::enable_if<!has_member_function_gather<TRes *, sp::spPage *, Args...>::value>::type * = nullptr>
    void gather(TRes *res, mesh::point_type const &x0, Args &&...args) const
    {
        *res = 0;

        mesh::MeshEntityId s = std::get<0>(this->m_mesh_->point_global_to_local(x0));

        mesh::MeshEntityId neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

        int num = this->m_mesh_->get_adjacent_entities(this->entity_type(), s, neighbours);

        for (int i = 0; i < num; ++i)
        {
            SP_PAGE_FOREACH(value_type, p, this->get(neighbours[i]))
            {
                engine_type::gather(res, *p, x0, std::forward<Args>(args) ...);
            }
        }
    }

    template<typename TV, typename ...Others, typename ...Args>
    void gather(Field<TV, mesh_type, Others...> *res, range_type const &, Args &&...args) const;

    template<typename TV, typename ...Others, typename ...Args>
    void gather_all(Field<TV, mesh_type, Others...> *res, Args &&...args) const;


    template<typename ...Args,
            typename std::enable_if<!has_member_function_push<engine_type, sp::spPage *, Args...>::value>::type * = nullptr>
    void push(range_type const &r0, Args &&...args)
    {
        parallel::parallel_for(
                r0,
                [&](range_type const &r)
                {
                    for (auto const &s: r)
                    {
                        sp::spPage *pg = this->get(s);

                        SP_PAGE_FOREACH(value_type, p, pg)
                        {
                            engine_type::push(p, std::forward<Args>(args) ...);
                        }

                    }
                });
    };

    template<typename ...Args,
            typename std::enable_if<has_member_function_push<engine_type, sp::spPage *, Args...>::value>::type * = nullptr>
    void push(range_type const &r0, Args &&...args)
    {
        parallel::parallel_for(r0, [&](range_type const &r)
        {
            for (auto const &s: r)
            {
                engine_type::push(this->get(s), std::forward<Args>(args) ...);
            }
        });
    };

    template<typename ...Args> void push_all(Args &&...args);

    void clear_duplicates(range_type const &r);

    void neighbour_resort(range_type const &r);

    void neighbour_resort();

    //**************************************************************************************************
    //! @name as container
    //! @{
    void insert(id_type const &s, value_type const &v);

    template<typename TInputIterator>
    void insert(id_type const &s, TInputIterator, TInputIterator);


    template<typename Predicate>
    void remove_if(id_type const &s, Predicate const &pred);

    template<typename Predicate>
    void remove_if(range_type const &r, Predicate const &pred);

    void erase(id_type const &s);

    void erase(range_type const &r);

    size_t count(range_type const &r) const;

    size_t count() const { return count(entity_id_range()); };

    template<typename TFun, typename ...Args>
    void apply(range_type const &r, TFun const &op, Args &&...);

    template<typename TFun, typename ...Args>
    void apply(range_type const &r, TFun const &op, Args &&...) const;

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&...args)
    {
        apply(field_type::m_mesh_->range(field_type::entity_type()), op, std::forward<Args>(args)...);
    };

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&...args) const
    {
        apply(field_type::m_mesh_->range(field_type::entity_type()), op, std::forward<Args>(args)...);
    };


};//class Particle


template<typename P, typename M>
void
Particle<P, M, V001>::clear()
{
    deploy();
    field_type::clear();
}


template<typename P, typename M>
bool
Particle<P, M, V001>::deploy()
{
    bool success = true;
    success = success && field_type::deploy();
    engine_type::deploy();

//
//    if (m_holder_ == nullptr)
//    {
//        if (m_data_ == nullptr)
//        {
//            if (m_mesh_ == nullptr)
//            {
//                RUNTIME_ERROR << "get_mesh is not valid!" <<
//                std::endl;
//            }
//            else
//            {
//                m_data_ = std::make_shared<container_type>();
//                m_properties_ = std::make_shared<Properties>();
//                sp::makePagePool(sizeof(value_type)).swap(m_pool_);
//            }
//
//            success = true;
//        }
//    }
//    else
//    {
//        if (m_holder_->is_a<this_type>())
//        {
//            m_holder_->deploy();
//
//            auto self = std::dynamic_pointer_cast<this_type>(m_holder_);
//            m_mesh_ = self->m_mesh_;
//            m_data_ = self->m_data_;
//            m_properties_ = self->m_properties_;
//            m_pool_ = self->m_pool_;
//            success = true;
//        }
//    }


    return success;

}
//**************************************************************************************************

template<typename P, typename M>
std::ostream &Particle<P, M, V001>::print(std::ostream &os, int indent) const
{
//    os << std::setw(indent + 1) << "   Type=" << class_name() << " , " << std::endl;
    os << std::setw(indent + 1) << " num = " << count() << " , ";

    properties().print(os, indent + 1);
    return os;
}



//**************************************************************************************************

template<typename P, typename M>
template<typename TFun, typename ...Args>
void
Particle<P, M, V001>::apply(range_type const &r0, TFun const &op, Args &&...args)
{
    parallel::parallel_for(r0, [&](range_type const &r)
    {
//        typename container_type::accessor acc;
//        for (auto const &s: r)
//        {
//            if (m_data_->find(acc, s))
//            {
//                for (sp::spIterator __it = {0x0, 0x0, acc->second, sizeof(value_type)};
//                     sp::spItTraverse(&__it) != 0x0;)
//                {
//                    fun(reinterpret_cast<value_type *>(__it.p), std::forward<Args>(args) ...);
//                }
//            }
//            acc.release();
//        }
    });
}

template<typename P, typename M>
template<typename TFun, typename ...Args>
void
Particle<P, M, V001>::apply(range_type const &r0, TFun const &op, Args &&...args) const
{
    parallel::parallel_for(
            r0,
            [&](range_type const &r)
            {
//                typename container_type::const_accessor acc;
//                for (auto const &s: r)
//                {
//                    if (m_data_->find(acc, s))
//                    {
//                        for (sp::spIterator __it = {0x0, 0x0, acc->second, sizeof(value_type)};
//                             sp::spItTraverse(&__it) != 0x0;)
//                        {
//                            fun(*reinterpret_cast<value_type *>(__it.p), std::forward<Args>(args) ...);
//                        }
//                    }
//                    acc.release();
//                }
            });
}

//**************************************************************************************************

template<typename P, typename M>
void
Particle<P, M, V001>::erase(id_type const &key) { }

template<typename P, typename M>
void
Particle<P, M, V001>::erase(range_type const &r) { for (auto const &s:r) { erase(s); }}
//**************************************************************************************************

//**************************************************************************************************


template<typename P, typename M>
data_model::DataSet
Particle<P, M, V001>::dataset(range_type const &r0) const
{
    data_model::DataSet ds;

    size_t num = count(r0);
    if (num > 0)
    {
        ds.data_type = data_model::DataType::create<value_type>();

        ds.data = sp_alloc_memory(num * sizeof(value_type));

//    ds.properties = this->properties();

        std::tie(ds.data_space, ds.memory_space) = data_model::DataSpace::create_simple_unordered(num);

//        copy(r0, reinterpret_cast< value_type *>( ds.data.get()));
    }
    return std::move(ds);
};

template<typename P, typename M> void
Particle<P, M, V001>::dataset(data_model::DataSet const &d)
{
    dataset(field_type::m_mesh_->range(field_type::entity_type()), d);
};


template<typename P, typename M> void
Particle<P, M, V001>::dataset(mesh::MeshEntityRange const &r0, data_model::DataSet const &ds)
{


//    ptrdiff_t inc = std::max(10, (num_of_element / r.size()));
//    std::map<id_type, std::tuple<size_t, void *>> ptr;
//
//    {
//        auto it = r0.begin();
//
//        auto ie = r0.end();
//
//
//
//        while ((num_of_element > 0) && (it != ie))
//        {
//            size_t n = std::min(inc, num_of_element);
//
//            ptr.emplace(std::make_pair(*it, std::make_tuple(n, v));
//
//            v += n * size_in_byte;
//            num_of_element -= n;
//            ++it;
//        }
//    }



};


//**************************************************************************************************

template<typename P, typename M>
size_t
Particle<P, M, V001>::count(range_type const &r0) const
{

    return parallel::parallel_reduce(
            r0, 0U,
            [&](range_type const &r, size_t init) -> size_t
            {
//                for (auto const &s:r)
//                {
//                    typename container_type::const_accessor acc;
//
//                    if (m_data_->find(acc, s)) { init += sp::spSize(acc->second); }
//                }

                return init;
            },
            [](size_t x, size_t y) -> size_t { return x + y; }
    );
}

//**************************************************************************************************

//
//
//template<typename P, typename M>
//template<typename OutputIT>
//OutputIT
//Particle<P, M, V001>::copy(range_type const &r, OutputIT out_it) const
//{
//    //TODO need optimize
//    for (auto const &s:r) { out_it = copy(s, out_it); }
//    return out_it;
//}


//*******************************************************************************


//template<typename P, typename M>
//void
//Particle<P, M, V001>::merge(buffer_type *buffer)
//{
//    parallel::parallel_for(
//            buffer->range(),
//            [&](typename buffer_type::range_type const &r)
//            {
//                for (auto const &item:r)
//                {
//                    typename container_type::accessor acc1;
//                    m_data_->insert(acc1, item.first);
//
//                    auto *p = acc1->second;
//                    acc1->second = item.second;
//                    acc1->second->next = p;
//
//                }
//
//            }
//    );
//}


template<typename P, typename M> template<typename TInputIterator> void
Particle<P, M, V001>::insert(id_type const &s,
                             TInputIterator ib, TInputIterator ie)
{
//    typename container_type::accessor acc;
//    std::mutex m_pool_mutex_;
//    data->insert(acc, s);
//
//    while (ib != ie)
//    {
//        for (sp::spIterator __it = {0x0, 0x0, acc->second, sizeof(value_type)};
//             sp::spItInsert(&__it) != 0x0 && (ib != ie); ++ib)
//        {
//            *reinterpret_cast<value_type *>(__it.p) = *ib;
//        }
//        if (ib != ie)
//        {
//            //FIXME  here need atomic op
//            std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
//            sp::spPage *p = acc->second;
//            acc->second = sp::spPageCreate(m_pool_.get());
//            acc->second->next = p;
//        }
//    }
}

//template<typename P, typename M> void
//Particle<P, M, V001>::insert(id_type const &s, value_type const &v) { insert(s, &v, &v + 1); }
//
//
//template<typename P, typename M> template<typename Hash, typename TRange> void
//Particle<P, M, V001>::insert(Hash const &hash, TRange const &v_r)
//{
//    parallel::parallel_for(v_r, [&](TRange const &r) { for (auto const &p: v_r) { insert(hash(p), p); }});
//};
//
//template<typename P, typename M>
//template<typename TInputIterator>
//void
//Particle<P, M, V001>::insert(id_type const &s, TInputIterator ib, TInputIterator ie)
//{
////    _insert(m_data_.get(), s, ib, ie);
//}

//*******************************************************************************

template<typename P, typename M>
template<typename Predicate>
void
Particle<P, M, V001>::remove_if(id_type const &s, Predicate const &pred)
{
//    typename container_type::accessor acc;
//
//    if (m_data_->find(acc, s))
//    {
//        int flag = 0;
//        for (sp::spIterator __it = {0x0, 0x0, acc->second, sizeof(value_type)};
//             sp::spItRemoveIf(&__it, flag) != 0x0;)
//        {
//            flag = pred(reinterpret_cast<value_type *>(__it.p)) ? 1 : 0;
//        }
//    }
}


template<typename P, typename M>
template<typename Predicate>
void
Particle<P, M, V001>::remove_if(range_type const &r0, Predicate const &pred)
{
    parallel::parallel_for(r0, [&](range_type const &r) { for (auto const &s:r) { remove_if(s, pred); }});
}


//*******************************************************************************


template<typename P, typename M> void
Particle<P, M, V001>::clear_duplicates(range_type const &r)
{

};

template<typename P, typename M> void
Particle<P, M, V001>::neighbour_resort()
{
    neighbour_resort(entity_id_range());
}


template<typename P, typename M> void
Particle<P, M, V001>::neighbour_resort(range_type const &r)
{
    parallel::parallel_for(r, [&](range_type const &r)
    {
        for (auto const &key: r)
        {

            size_t number_of_neighbours = this->m_mesh_->get_adjacent_entities(entity_type(), key);

            mesh::MeshEntityId neighbour_ids[number_of_neighbours];

            this->m_mesh_->get_adjacent_entities(entity_type(), key, neighbour_ids);

            sp::spPage *neighbour[number_of_neighbours];

            for (int s = 0; s < number_of_neighbours; ++s)
            {
                neighbour[s] = field_type::operator[](s);
            }

            sp::spPage **self = &(*this)[key];

            sp::spPage **buffer;

//            {
//                std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
//                sp::spPage *t = sp::spPageCreateN(5, m_pool_.get());
//            }
//
//            sp::PushFront(&sp::spPageCreateN(5, m_pool_.get()), buffer);
//
//            sp::spSetTag(*buffer, 0x0);
//
//            spParticleCopyN(key0, sizeof(value_type), number_of_neighbours, neighbour, self, &buffer);
//            spParticleClear(key0, self);
//
//            {
//                std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
//                sp::Merge(buffer, &(m_pool_.get()));
//            }
        }
    });
};


template<typename P, typename M> template<typename ...Args> void
Particle<P, M, V001>::push_all(Args &&...args)
{
    if (!this->is_valid()) { RUNTIME_ERROR << "Particle is not valid! [" << get_class_name() << "]" << std::endl; }
    logger::Logger __logger(logger::LOG_VERBOSE);
    __logger << "CMD:\t" << "Push   [" << get_class_name() << "]";

/**
 *   |<-----------------------------     valid   --------------------------------->|
 *   |<- not owned  ->|<-------------------       owned     ---------------------->|
 *   |----------------*----------------*---*---------------------------------------|
 *   |<---- ghost --->|                |   |                                       |
 *   |<------------ shared  ---------->|<--+--------  not shared  ---------------->|
 *   |<------------- DMZ    -------------->|<----------   not DMZ   -------------->|
 *
 */
    push(entity_id_range(DMZ), std::forward<Args>(args)...);
    neighbour_resort(entity_id_range(DMZ));
    clear_duplicates(entity_id_range(SHARED));
    base_type::nonblocking_sync();
    push(entity_id_range(NOT_DMZ), std::forward<Args>(args)...);
    neighbour_resort(entity_id_range(NOT_SHARED));
    clear_duplicates(entity_id_range(NOT_SHARED));
    base_type::wait();

    __logger << DONE;

}

//**************************************************************************************************


template<typename P, typename M> template<typename TV, typename ...Others, typename ...Args> void
Particle<P, M, V001>::gather_all(Field<TV, mesh_type, Others...> *res, Args &&...args) const
{
    if (!this->is_valid()) { RUNTIME_ERROR << "Particle is not valid! [" << get_class_name() << "]" << std::endl; }
    logger::Logger __logger(logger::LOG_VERBOSE);
    __logger << "CMD:\t" << "Gather   [" << get_class_name() << "]";
    typedef typename traits::field_value_type<Field<TV, mesh_type, Others...>>::type field_value_type;

    res->apply(
            [&](point_type const &x) -> field_value_type
            {
                field_value_type v;
                this->gather(&v, x, std::forward<Args>(args)...);
                return v;
            });

    __logger << DONE;
};



//**************************************************************************************************
}}//namespace simpla { namespace particle
#endif //SIMPLA_PARTICLELITE_H
