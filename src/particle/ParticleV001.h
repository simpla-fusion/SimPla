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

namespace simpla { namespace particle
{

template<typename ...>
struct Particle;
typedef typename simpla::tags::VERSION<0, 0, 1> V001;


template<typename P, typename M>
struct Particle<P, M, V001>
        : public mesh::MeshAttribute::View, public P,
          public std::enable_shared_from_this<Particle<P, M, V001>>
{
private:

    typedef Particle<P, M, V001> this_type;
    typedef mesh::MeshAttribute::View base_type;
public:

    typedef M mesh_type;
    typedef P engine_type;
    typedef typename P::point_s value_type;
    typedef std::shared_ptr<struct sp::spPage> bucket_type;
    typedef std::shared_ptr<struct sp::spPagePool> pool_type;
    typedef typename mesh::MeshEntityId id_type;
    typedef typename mesh::MeshEntityRange range_type;
    typedef parallel::concurrent_hash_map<id_type, bucket_type> container_type;
    typedef container_type buffer_type;

private:
    mesh_type const *m_mesh_;
    pool_type m_pool_;
    std::mutex m_pool_mutex_;
    std::shared_ptr<container_type> m_data_;
    std::shared_ptr<base_type> m_holder_;
    static constexpr mesh::MeshEntityType iform = mesh::VOLUME;

public:
    virtual Properties const &properties() const
    {
        assert(m_properties_ != nullptr);
        return *m_properties_;
    };


    virtual Properties &properties()
    {
        assert(m_properties_ != nullptr);
        return *m_properties_;
    };

private:
    std::shared_ptr<Properties> m_properties_;
public:


    Particle(mesh_type const *m = nullptr)
            : m_holder_(nullptr), m_mesh_(m), m_data_(nullptr), m_properties_(nullptr),
              m_pool_(nullptr)
    {
    }

    Particle(mesh::MeshBase const *m)
            : m_holder_(nullptr), m_mesh_(dynamic_cast<mesh_type const *>(m)),
              m_data_(nullptr), m_properties_(nullptr), m_pool_(nullptr)
    {
        assert(m->template is_a<mesh_type>());
    }

    Particle(std::shared_ptr<base_type> h)
            : m_holder_(h), m_mesh_(nullptr), m_data_(nullptr), m_pool_(nullptr)
    {
        deploy();
    }

    //factory construct
    template<typename TFactory, typename ... Args, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
    Particle(TFactory &factory, Args &&...args)
            : m_holder_(std::dynamic_pointer_cast<base_type>(
            factory.template create<this_type>(std::forward<Args>(args)...))),
              m_mesh_(nullptr), m_data_(nullptr), m_properties_(nullptr), m_pool_(nullptr)
    {
        deploy();
    }


    //copy construct
    Particle(this_type const &other)
            : engine_type(other), m_holder_(other.m_holder_), m_mesh_(other.m_mesh_),
              m_data_(other.m_data_), m_properties_(other.m_properties_), m_pool_(other.m_pool_)
    {
    }


    // move construct
    Particle(this_type &&other)
            : engine_type(other), m_holder_(other.m_holder_), m_mesh_(other.m_mesh_),
              m_data_(other.m_data_), m_properties_(other.m_properties_), m_pool_(other.m_pool_)
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
        std::swap(other.m_mesh_, m_mesh_);
        std::swap(other.m_data_, m_data_);
        std::swap(other.m_holder_, m_holder_);
        std::swap(m_properties_, other.m_properties_);
        std::swap(m_pool_, other.m_pool_);

    }

    virtual mesh::MeshBase const *get_mesh() const { return dynamic_cast<mesh::MeshBase const *>(m_mesh_); };

    virtual bool set_mesh(mesh::MeshBase const *m)
    {
        UNIMPLEMENTED;
        assert(m->is_a<mesh_type>());
        m_mesh_ = dynamic_cast<mesh_type const * >(m);
        return false;
    }

    virtual mesh::MeshEntityRange entity_id_range() const { return m_mesh_->range(entity_type()); }

    container_type &data() { return *m_data_; }

    container_type const &data() const { return *m_data_; }

    std::ostream &print(std::ostream &os, int indent) const;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); };

    virtual std::string get_class_name() const { return class_name(); };

    static std::string class_name() { return "Particle<" + traits::type_id<P, M>::name() + ">"; };

    virtual bool is_valid() const { return m_data_ != nullptr && m_mesh_ != nullptr; }

    virtual bool deploy();

    virtual mesh::MeshEntityType entity_type() const { return iform; }


    virtual data_model::DataSet dataset(range_type const &) const;

    virtual data_model::DataSet dataset() const { return dataset(m_mesh_->range(entity_type())); }

    virtual void dataset(data_model::DataSet const &);

    virtual void dataset(mesh::MeshEntityRange const &, data_model::DataSet const &);


    virtual size_t size() const { return count(m_mesh_->range(entity_type())); }

    virtual void clear();

    template<typename TRes, typename ...Args>
    void gather(TRes *res, mesh::point_type const &x0, Args &&...args) const;

    template<typename TV, typename ...Others, typename ...Args>
    void gather(Field<TV, mesh_type, Others...> *res, Args &&...args) const;

    template<typename ...Args>
    void push(Args &&...args);


    template<typename TFun, typename ...Args>
    void apply(range_type const &r, TFun const &op, Args &&...);

    template<typename TFun, typename ...Args>
    void apply(range_type const &r, TFun const &op, Args &&...) const;

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&...args)
    {
        apply(m_mesh_->range(entity_type()), op, std::forward<Args>(args)...);
    };

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&...args) const
    {
        apply(m_mesh_->range(entity_type()), op, std::forward<Args>(args)...);
    };

    //**************************************************************************************************
    //! @name as container
    //! @{
    void insert(id_type const &s, value_type const &v);

    template<typename TInputIterator>
    void insert(id_type const &s, TInputIterator, TInputIterator);

    template<typename Hash, typename TRange>
    void insert(Hash const &, TRange const &);

private    :
    template<typename TInputIterator>
    void _insert(container_type *m_data_, id_type const &s, TInputIterator ib, TInputIterator ie);

public:
    template<typename Predicate>
    void remove_if(id_type const &s, Predicate const &pred);

    template<typename Predicate>
    void remove_if(range_type const &r, Predicate const &pred);

    void erase(id_type const &s);

    void erase(range_type const &r);

    template<typename THash>
    void rehash(id_type const &key, THash const &hash, buffer_type *out_buffer);

    template<typename THash>
    void rehash(range_type const &key, THash const &hash, buffer_type *out_buffer);

    size_t count(range_type const &r) const;

    size_t count() const { return count(m_mesh_->range(entity_type())); };

    template<typename OutputIT>
    OutputIT copy(id_type const &s, OutputIT out_it) const;

    template<typename OutputIT>
    OutputIT copy(range_type const &r, OutputIT out_it) const;

    void merge(buffer_type *other);

};//class Particle


template<typename P, typename M>
void
Particle<P, M, V001>::clear()
{
    deploy();
    m_data_->clear();
}


template<typename P, typename M>
bool
Particle<P, M, V001>::deploy()
{
    bool success = false;

    if (m_holder_ == nullptr)
    {
        if (m_data_ == nullptr)
        {
            if (m_mesh_ == nullptr)
            {
                RUNTIME_ERROR << "get_mesh is not valid!" <<
                std::endl;
            }
            else
            {
                m_data_ = std::make_shared<container_type>();
                m_properties_ = std::make_shared<Properties>();
                sp::makePagePool(sizeof(value_type)).swap(m_pool_);
            }

            success = true;
        }
    }
    else
    {
        if (m_holder_->is_a<this_type>())
        {
            m_holder_->deploy();

            auto self = std::dynamic_pointer_cast<this_type>(m_holder_);
            m_mesh_ = self->m_mesh_;
            m_data_ = self->m_data_;
            m_properties_ = self->m_properties_;
            m_pool_ = self->m_pool_;
            success = true;
        }
    }

    engine_type::deploy();

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
template<typename TRes, typename ...Args>
void
Particle<P, M, V001>::gather(TRes *res, mesh::point_type const &x0, Args &&...args) const
{
    *res = 0;
    mesh::MeshEntityId s = std::get<0>(m_mesh_->point_global_to_local(x0));

    mesh::MeshEntityId neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = m_mesh_->get_adjacent_entities(entity_type(), s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (m_data_->find(acc1, neighbours[i]))
        {

            for (sp::spIterator __it = {0x0, 0x0, acc1->second.get(), sizeof(value_type)};
                 sp::spTraverseIterator(&__it) != 0x0;)
            {
                engine_type::gather(res, *reinterpret_cast<value_type *>(__it.p), x0, std::forward<Args>(args) ...);
            }
        }
    }

}

template<typename P, typename M>
template<typename TV, typename ...Others, typename ...Args>
void
Particle<P, M, V001>::gather(Field<TV, mesh_type, Others...> *res, Args &&...args) const
{

//FIXME  using this->box() select entity_id_range
    if (is_valid())
    {
        LOGGER << "Gather [" << get_class_name() << "]" << std::endl;
        res->apply(res->entity_id_range(),
                   [&](point_type const &x)
                   {
                       typename traits::field_value_type<Field<TV, mesh_type, Others...>>::type v;
                       this->gather(&v, x, std::forward<Args>(args)                               ...);
                       return v;
                   });
        res->sync();

    }
    else
    {
        LOGGER << "Particle is not valid! [" << get_class_name() << "]" << std::endl;
    }
};

template<typename P, typename M>
template<typename ...Args>
void
Particle<P, M, V001>::push(Args &&...args)
{
    if (is_valid())
    {
        LOGGER << "Push   [" << get_class_name() << "]" << std::endl;

        parallel::parallel_for(
                m_mesh_->range(entity_type()),
                [&](range_type const &r)
                {
                    typename container_type::accessor acc;
                    for (auto const &s: r)
                    {
                        if (m_data_->find(acc, s))
                        {
                            for (sp::spIterator __it = {0x0, 0x0, acc->second.get(), sizeof(value_type)};
                                 sp::spTraverseIterator(&__it) != 0x0;)
                            {
                                engine_type::push(reinterpret_cast<value_type *>(__it.p), std::forward<Args>(args) ...);
                            }
                        }
                        acc.release();
                    }
                });

    }
    else
    {
        LOGGER << "Particle is not valid! [" << get_class_name() << "]" << std::endl;
    }
}
//**************************************************************************************************

template<typename P, typename M>
template<typename TFun, typename ...Args>
void
Particle<P, M, V001>::apply(range_type const &r0, TFun const &op, Args &&...args)
{
    parallel::parallel_for(r0, [&](range_type const &r)
    {
        typename container_type::accessor acc;
        for (auto const &s: r)
        {
            if (m_data_->find(acc, s))
            {
                for (sp::spIterator __it = {0x0, 0x0, acc->second.get(), sizeof(value_type)};
                     sp::spTraverseIterator(&__it) != 0x0;)
                {
                    fun(reinterpret_cast<value_type *>(__it.p), std::forward<Args>(args) ...);
                }
            }
            acc.release();
        }
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
                typename container_type::const_accessor acc;
                for (auto const &s: r)
                {
                    if (m_data_->find(acc, s))
                    {
                        for (sp::spIterator __it = {0x0, 0x0, acc->second.get(), sizeof(value_type)};
                             sp::spTraverseIterator(&__it) != 0x0;)
                        {
                            fun(*reinterpret_cast<value_type *>(__it.p), std::forward<Args>(args) ...);
                        }
                    }
                    acc.release();
                }
            });
}

//**************************************************************************************************

template<typename P, typename M>
void
Particle<P, M, V001>::erase(id_type const &key) { m_data_->erase(key); }

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

        copy(r0, reinterpret_cast< value_type *>( ds.data.get()));
    }
    return std::move(ds);
};

template<typename P, typename M> void
Particle<P, M, V001>::dataset(data_model::DataSet const &)
{
    UNIMPLEMENTED;
};

template<typename P, typename M> void
Particle<P, M, V001>::dataset(mesh::MeshEntityRange const &, data_model::DataSet const &ds)
{
    value_type const *src = reinterpret_cast<value_type const *>(ds.data.get());
    size_t count = ds.memory_space.num_of_elements();
    sp::spPage *pg = sp::spPageCreate(m_pool_.get());

    while (count > 0)
    {
        for (sp::spIterator __it = {0x0, 0x0, pg, sizeof(value_type)};
             sp::spInsertIterator(&__it) != 0x0 && (count != 0); --count, ++src)
        {
            *reinterpret_cast<value_type *>(__it.p) = *src;
        }
        if (count != 0)
        {
            std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
            sp::spPage *p = pg;
            pg = sp::spPageCreate(m_pool_.get());
            pg->next = p;
        }
    }
    std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
    sp::spPageClose(pg, m_pool_.get());
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
                for (auto const &s:r)
                {
                    typename container_type::const_accessor acc;

                    if (m_data_->find(acc, s)) { init += sp::spPageCount(acc->second.get()); }
                }

                return init;
            },
            [](size_t x, size_t y) -> size_t { return x + y; }
    );
}

//**************************************************************************************************


template<typename P, typename M>
template<typename OutputIterator>
OutputIterator
Particle<P, M, V001>::copy(id_type const &s, OutputIterator out_it) const
{
    typename container_type::const_accessor c_accessor;
    if (m_data_->find(c_accessor, s))
    {
        for (sp::spIterator __it = {0x0, 0x0, c_accessor->second.get(), sizeof(value_type)};
             sp::spTraverseIterator(&__it) != 0x0;)
        {
            *out_it = *reinterpret_cast<value_type *>(__it.p);
        }

    }
    return out_it;
}

template<typename P, typename M>
template<typename OutputIT>
OutputIT
Particle<P, M, V001>::copy(range_type const &r, OutputIT out_it) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy(s, out_it); }
    return out_it;
}


//*******************************************************************************


template<typename P, typename M>
void
Particle<P, M, V001>::merge(buffer_type *buffer)
{
    parallel::parallel_for(
            buffer->range(),
            [&](typename buffer_type::range_type const &r)
            {
                for (auto const &item:r)
                {
                    typename container_type::accessor acc1;
                    m_data_->insert(acc1, item.first);

                    auto *p = acc1->second.get();
                    acc1->second = item.second;
                    acc1->second->next = p;

                }

            }
    );
}


template<typename P, typename M> template<typename TInputIterator> void
Particle<P, M, V001>::_insert(container_type *data, id_type const &s,
                              TInputIterator ib, TInputIterator ie)
{
    typename container_type::accessor acc;

    data->insert(acc, s);

    while (ib != ie)
    {
        for (sp::spIterator __it = {0x0, 0x0, acc->second.get(), sizeof(value_type)};
             sp::spInsertIterator(&__it) != 0x0 && (ib != ie); ++ib)
        {
            *reinterpret_cast<value_type *>(__it.p) = *ib;
        }
        if (ib != ie)
        {
            //FIXME  here need atomic op
            std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
            sp::spPage *p = acc->second.get();
            sp::makePage(m_pool_).swap(acc->second);
            acc->second->next = p;
        }
    }
}

template<typename P, typename M> void
Particle<P, M, V001>::insert(id_type const &s, value_type const &v) { insert(s, &v, &v + 1); }


template<typename P, typename M> template<typename Hash, typename TRange> void
Particle<P, M, V001>::insert(Hash const &hash, TRange const &v_r)
{
    parallel::parallel_for(v_r, [&](TRange const &r) { for (auto const &p: v_r) { insert(hash(p), p); }});
};

template<typename P, typename M>
template<typename TInputIterator>
void
Particle<P, M, V001>::insert(id_type const &s, TInputIterator ib, TInputIterator ie)
{
    _insert(m_data_.get(), s, ib, ie);
}

//*******************************************************************************

template<typename P, typename M>
template<typename Predicate>
void
Particle<P, M, V001>::remove_if(id_type const &s, Predicate const &pred)
{
    typename container_type::accessor acc;

    if (m_data_->find(acc, s))
    {
        int flag = 0;
        for (sp::spIterator __it = {0x0, 0x0, acc->second.get(), sizeof(value_type)};
             sp::spRemoveIfIterator(&__it, flag) != 0x0;)
        {
            flag = pred(reinterpret_cast<value_type *>(__it.p)) ? 1 : 0;
        }
    }
}


template<typename P, typename M>
template<typename Predicate>
void
Particle<P, M, V001>::remove_if(range_type const &r0, Predicate const &pred)
{
    parallel::parallel_for(r0, [&](range_type const &r) { for (auto const &s:r) { remove_if(s, pred); }});
}



//*******************************************************************************

template<typename P, typename M> template<typename THash> void
Particle<P, M, V001>::rehash(id_type const &key0, THash const &hash, buffer_type *out_buffer)
{
    assert(out_buffer != nullptr);

    typename buffer_type::accessor acc0;

    if (m_data_->find(acc0, key0))
    {
        remove_if(key0, [&](value_type const &p)
        {
            auto key1 = hash(p);
            if (key1 != key0)
            {
                std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
                _insert(out_buffer, key1, &p, &p + 1);
                return true;
            }
            else
            {
                return false;
            }
        });


    }

    acc0.release();

};

template<typename P, typename M>
template<typename THash>
void
Particle<P, M, V001>::rehash(range_type const &r, THash const &hash, buffer_type *out_buffer)
{
    parallel::parallel_for(r, [&](range_type const &r) { for (auto const &s: r) { rehash(s, hash, out_buffer); }});
};

//**************************************************************************************************
}}//namespace simpla { namespace particle
#endif //SIMPLA_PARTICLELITE_H