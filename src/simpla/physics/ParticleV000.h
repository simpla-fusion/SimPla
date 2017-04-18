//
// Created by salmon on 16-6-7.
//

#ifndef SIMPLA_PARTICLEV000_H
#define SIMPLA_PARTICLEV000_H

#include <vector>
#include <list>
#include <map>

#include "../toolbox/Parallel.h"
#include "simpla/mesh/AttributeView.h"

namespace simpla
{
namespace particle
{


template<typename ...> struct Particle;

template<typename P, typename M>
struct Particle<P, M>
        : public mesh::AttributeDesc::View, public P,
          public std::enable_shared_from_this<Particle<P, M>>
{
private:

    typedef Particle<P, M> this_type;
    typedef mesh::AttributeDesc::View base_type;
public:

    typedef M mesh_type;
    typedef P engine_type;
    typedef typename P::point_s value_type;
    typedef std::list<value_type> bucket_type;
    typedef typename EntityId id_type;
    typedef typename mesh::EntityRange range_type;
    typedef parallel::concurrent_hash_map<id_type, bucket_type> container_type;
    typedef container_type buffer_type;

private:
    mesh_type const *m_mesh_;

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
            : m_holder_(nullptr), m_mesh_(m), m_data_(nullptr), m_properties_(new Properties), MeshAttribute(nullptr),
              MeshAttribute(nullptr), AttributeDesc(<#initializer#>, 0, 0, 0, <#initializer#>) {
    }

    Particle(mesh::MeshView const *m)
            : m_holder_(nullptr), m_mesh_(dynamic_cast<mesh_type const *>(m)),
              m_data_(nullptr), m_properties_(new Properties), MeshAttribute(nullptr), MeshAttribute(nullptr),
              AttributeDesc(<#initializer#>, 0, 0, 0, <#initializer#>) {
        assert(m->template is_a<mesh_type>());
    }

    Particle(std::shared_ptr<base_type> h)
            : m_holder_(h), m_mesh_(nullptr), m_data_(nullptr), MeshAttribute(nullptr), MeshAttribute(nullptr),
              AttributeDesc(<#initializer#>, 0, 0, 0, <#initializer#>) {
        deploy();
    }

    //factory construct
    template<typename TFactory, typename ... Args, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
    Particle(TFactory &factory, Args &&...args)
            : m_holder_(std::dynamic_pointer_cast<base_type>(
            factory.template create<this_type>(std::forward<Args>(args)...))),
              m_mesh_(nullptr), m_data_(nullptr), m_properties_(nullptr), MeshAttribute(nullptr), MeshAttribute(nullptr),
              AttributeDesc(<#initializer#>, 0, 0, 0, <#initializer#>) {
        deploy();
    }


    //Duplicate construct
    Particle(this_type const &other)
            : engine_type(other), m_holder_(other.m_holder_), m_mesh_(other.m_mesh_),
              m_data_(other.m_data_), m_properties_(other.m_properties_), MeshAttribute(nullptr), MeshAttribute(nullptr),
              AttributeDesc(<#initializer#>, 0, 0, 0, <#initializer#>) {
    }


    // Move construct
    Particle(this_type &&other)
            : engine_type(other), m_holder_(other.m_holder_), m_mesh_(other.m_mesh_),
              m_data_(other.m_data_), m_properties_(other.m_properties_), MeshAttribute(nullptr), MeshAttribute(nullptr),
              AttributeDesc(<#initializer#>, 0, 0, 0, <#initializer#>) {
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
    }

    virtual mesh::MeshView const *get_mesh() const { return dynamic_cast<mesh::MeshView const *>(m_mesh_); };

    virtual bool set_mesh(mesh::MeshView const *m)
    {
        UNIMPLEMENTED;
        assert(m->is_a<mesh_type>());
        m_mesh_ = dynamic_cast<mesh_type const * >(m);
        return false;
    }

    virtual mesh::EntityRange entity_id_range() const { return m_mesh_->range(entity_type()); }

    container_type &data() { return *m_data_; }

    container_type const &data() const { return *m_data_; }

    std::ostream &print(std::ostream &os, int indent) const;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); };

    virtual std::string getClassName() const { return class_name(); };

    static std::string class_name() { return "Particle<" + traits::type_id<P, M>::name() + ">"; };

    virtual bool is_valid() const { return m_data_ != nullptr && m_mesh_ != nullptr; }

    virtual bool deploy();

    virtual mesh::MeshEntityType entity_type() const { return iform; }


    virtual data_model::DataSet dataset(range_type const &) const;

    virtual data_model::DataSet dataset() const { return dataset(m_mesh_->range(entity_type())); }

    virtual void dataset(data_model::DataSet const &);

    virtual void dataset(mesh::EntityRange const &, data_model::DataSet const &) { UNIMPLEMENTED; }


    virtual size_t size() const { return count(m_mesh_->range(entity_type())); }

    virtual void clear();

    virtual void clear(range_type const &r);

    template<typename TRes, typename ...Args>
    void gather(TRes *res, mesh::point_type const &x0, Args &&...args) const;

    template<typename TV, typename ...Others, typename ...Args>
    void gather_all(Field<TV, mesh_type, Others...> *res, Args &&...args) const;

    template<typename ...Args> void push(Args &&...args);


    template<typename TFun, typename ...Args>
    void apply(range_type const &r, TFun const &op, Args &&...);

    template<typename TFun, typename ...Args>
    void apply(range_type const &r, TFun const &op, Args &&...) const;

    template<typename TOP, typename ...Args> void apply(TOP const &op, Args &&...args)
    {
        apply(this->entity_id_range(), op, std::forward<Args>(args)...);
    };

    template<typename TOP, typename ...Args> void apply(TOP const &op, Args &&...args) const
    {
        apply(this->entity_id_range(), op, std::forward<Args>(args)...);
    };

    //**************************************************************************************************
    //! @name as container
    //! @{
    void insert(id_type const &s, value_type const &v);

    template<typename TInputIterator> void insert(id_type const &s, TInputIterator, TInputIterator);

    template<typename Hash, typename TRange> void insert(Hash const &, TRange const &);

    template<typename Predicate> void remove_if(id_type const &s, Predicate const &pred);

    template<typename Predicate> void remove_if(range_type const &r, Predicate const &pred);

    void erase(id_type const &s, container_type *out_buffer = nullptr);

    void erase(range_type const &r, container_type *buffer = nullptr);

    template<typename TPred>
    void erase_if(id_type const &s, TPred const &pred, buffer_type *buffer = nullptr);

    template<typename TPred>
    void erase_if(range_type const &r, TPred const &pred, container_type *buffer = nullptr);

    template<typename THash>
    void rehash(id_type const &key, THash const &hash, buffer_type *out_buffer);

    template<typename THash>
    void rehash(range_type const &key, THash const &hash, buffer_type *out_buffer);

    size_t count(range_type const &r) const;

    size_t count() const { return count(m_mesh_->range(entity_type())); };

    template<typename OutputIT> OutputIT copy(id_type const &s, OutputIT out_it) const;

    template<typename OutputIT> OutputIT copy(range_type const &r, OutputIT out_it) const;

    void merge(buffer_type *other);

};//class Particle


template<typename P, typename M> void
Particle<P, M>::clear()
{
    deploy();
    m_data_->clear();
}

template<typename P, typename M> void
Particle<P, M>::clear(range_type const &r)
{
    parallel::parallel_foreach([&](EntityId const &s) { m_data_->erase(s); }
}

);
};

template<typename P, typename M> bool
Particle<P, M>::deploy()
{
    bool success = false;

    if (m_holder_ == nullptr)
    {
        if (m_data_ == nullptr)
        {
            if (m_mesh_ == nullptr) { RUNTIME_ERROR << "get_mesh is not valid!" << std::endl; }
            else
            {
                m_data_ = std::make_shared<container_type>();
                m_properties_ = std::make_shared<Properties>();


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
            success = true;
        }
    }
    engine_type::deploy();
    return success;

}
//**************************************************************************************************

template<typename P, typename M>
std::ostream &Particle<P, M>::print(std::ostream &os, int indent) const
{
//    os << std::setw(indent + 1) << "   Type=" << class_name() << " , " << std::endl;
    os << std::setw(indent + 1) << " num = " << count() << " , ";
    properties().print(os, indent + 1);
    return os;
}

//**************************************************************************************************
template<typename P, typename M> template<typename TRes, typename ...Args> void
Particle<P, M>::gather(TRes *res, mesh::point_type const &x0, Args &&...args) const
{
    *res = 0;
    EntityId s = std::get<0>(m_mesh_->point_global_to_local(x0));

    EntityId neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

    int num = m_mesh_->get_adjacent_entities(entity_type(), s, neighbours);

    for (int i = 0; i < num; ++i)
    {
        typename container_type::const_accessor acc1;

        if (m_data_->find(acc1, neighbours[i]))
        {
            for (auto const &p:acc1->second)
            {
                engine_type::gather(res, p, x0, std::forward<Args>(args)...);
            }
        }
    }

}

template<typename P, typename M> template<typename TV, typename ...Others, typename ...Args> void
Particle<P, M>::gather_all(Field<TV, mesh_type, Others...> *res, Args &&...args) const
{

    //FIXME  using this->box() select entity_id_range
    if (is_valid())
    {
        LOGGER << "Gather [" << getClassName() << "]" << std::endl;

        res->apply(res->entity_id_range(),
                   [&](point_type const &x)
                   {

                       typename traits::field_value_type<Field < TV, mesh_type, Others...>>
                       ::type
                       v;
                       this->gather(&v, x, std::forward<Args>(args)...);
                       return v;
                   });
        res->sync();

    }
    else
    {
        LOGGER << "Particle is not valid! [" << getClassName() << "]" << std::endl;

    }
};

//**************************************************************************************************

template<typename P, typename M>
template<typename TFun, typename ...Args> void
Particle<P, M>::apply(range_type const &r0, TFun const &op, Args &&...args)
{
    parallel::parallel_for(r0, [&](range_type const &r)
    {
        typename container_type::accessor acc;
        for (auto const &s:r)
        {
            if (m_data_->find(acc, s))
            {
                for (auto &p:acc->second) { fun(&p, std::forward<Args>(args)...); }
            }
            acc.release();
        }
    });
//    for (auto const &s:r0) { for (auto &p:(*m_attr_data_)[s]) { op(&p); }}
}

template<typename P, typename M>
template<typename TFun, typename ...Args> void
Particle<P, M>::apply(range_type const &r0, TFun const &op, Args &&...args) const
{
    parallel::parallel_for(r0, [&](range_type const &r)
    {
        typename container_type::const_accessor acc;
        for (auto const &s:r)
        {
            if (m_data_->find(acc, s))
            {
                for (auto const &p:acc->second) { fun(p, std::forward<Args>(args)...); }
            }
            acc.release();
        }
    });
}

//**************************************************************************************************


template<typename P, typename M> void
Particle<P, M>::erase(id_type const &key, buffer_type *out_buffer)
{
    typename container_type::accessor acc1;
    if (m_data_->find(acc1, key))
    {
        if (out_buffer != nullptr)
        {
            typename container_type::accessor acc2;

            out_buffer->insert(acc2, key);

            acc2->second.splice(acc2->second.end(), acc1->second);

        }
        acc1->second.clear();
        acc1.release();
    }
};

template<typename P, typename M> void
Particle<P, M>::erase(range_type const &r, buffer_type *res)
{
    for (auto const &s:r) { erase(s, res); }
}
//**************************************************************************************************

template<typename P, typename M>
template<typename TPred> void
Particle<P, M>::erase_if(id_type const &key, TPred const &pred, buffer_type *out_buffer)
{

    typename buffer_type::accessor acc0;

    if (m_data_->find(acc0, key))
    {
        auto it = acc0->second.begin(), ie = acc0->second.end();

        while (it != ie)
        {
            auto p = it;
            ++it;
            if (pred(*p))
            {
                if (out_buffer != nullptr)
                {
                    typename container_type::accessor acc1;

                    out_buffer->insert(acc1, key);

                    acc1->second.splice(acc1->second.end(), acc0->second, it);

                } else
                {
                    acc0->second.erase(it);
                }
            }
        }
    }
    acc0.release();

}

template<typename P, typename M>
template<typename TPred> void
Particle<P, M>::erase_if(range_type const &r, TPred const &pred, buffer_type *out_buffer)
{
    for (auto const &s:r) { erase_if(s, pred, out_buffer); }
}
//**************************************************************************************************


template<typename P, typename M> data_model::DataSet
Particle<P, M>::dataset(range_type const &r0) const
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
Particle<P, M>::dataset(data_model::DataSet const &)
{
    UNIMPLEMENTED;
};

//**************************************************************************************************

template<typename V, typename K> size_t
Particle<V, K>::count(range_type const &r0) const
{

    return parallel::parallel_reduce(
            r0, 0U,
            [&](range_type const &r, size_t init) -> size_t
            {
                for (auto const &s:r)
                {
                    typename container_type::const_accessor acc;

                    if (m_data_->find(acc, s)) { init += acc->second.size(); }
                }

                return init;
            },
            [](size_t x, size_t y) -> size_t { return x + y; }
    );
}

//**************************************************************************************************


template<typename V, typename K>
template<typename OutputIterator> OutputIterator
Particle<V, K>::copy(id_type const &s, OutputIterator out_it) const
{
    typename container_type::const_accessor c_accessor;
    if (m_data_->find(c_accessor, s))
    {
        out_it = std::copy(c_accessor->second.begin(), c_accessor->second.end(), out_it);
    }
    return out_it;
}

template<typename V, typename K>
template<typename OutputIT> OutputIT
Particle<V, K>::copy(range_type const &r, OutputIT out_it) const
{
    //TODO need optimize
    for (auto const &s:r) { out_it = copy(s, out_it); }
    return out_it;
}


//*******************************************************************************


template<typename V, typename K> void
Particle<V, K>::merge(buffer_type *buffer)
{
    parallel::parallel_for(
            buffer->range(),
            [&](typename buffer_type::range_type const &r)
            {
                for (auto const &item:r)
                {
                    typename container_type::accessor acc1;
                    m_data_->insert(acc1, item.first);
                    acc1->second.splice(acc1->second.end(), item.second);
                }

            }
    );
}



//*******************************************************************************

template<typename P, typename M>
template<typename THash> void
Particle<P, M>::rehash(id_type const &key0, THash const &hash, buffer_type *out_buffer)
{
    assert(out_buffer != nullptr);

    typename buffer_type::accessor acc0;

    if (m_data_->find(acc0, key0))
    {
        auto it = acc0->second.begin(), ie = acc0->second.end();
        while (it != ie)
        {
            auto p = it;
            ++it;

            auto key1 = hash(*p);
            if (key1 != key0)
            {
                typename container_type::accessor acc1;
                out_buffer->insert(acc1, key1);
                acc1->second.splice(acc1->second.end(), acc0->second, p);
            }
        }
    }

    acc0.release();
};

template<typename P, typename M>
template<typename THash> void
Particle<P, M>::rehash(range_type const &r, THash const &hash, buffer_type *out_buffer)
{
    parallel::parallel_for(
            r,
            [&](range_type const &r)
            {
                for (auto const &s:r) { rehash(s, hash, out_buffer); }
            }
    );
};

//**************************************************************************************************
template<typename V, typename K> void
Particle<V, K>::insert(id_type const &s, value_type const &v)
{
    typename container_type::accessor acc;

    m_data_->insert(acc, s);

    acc->second.insert(acc->second.end(), v);
}

template<typename V, typename K> template<typename TInputIterator> void
Particle<V, K>::insert(id_type const &s, TInputIterator ib, TInputIterator ie)
{
    typename container_type::accessor acc;

    m_data_->insert(acc, s);

    acc->second.insert(acc->second.cend(), ib, ie);

}

template<typename V, typename K> template<typename Hash, typename TRange> void
Particle<V, K>::insert(Hash const &hash, TRange const &v_r)
{
    parallel::parallel_for(v_r, [&](TRange const &r)
    {
        for (auto const &p:v_r) { insert(hash(p), p); }
    });
};
//*******************************************************************************

template<typename V, typename K> template<typename Predicate> void
Particle<V, K>::remove_if(id_type const &s, Predicate const &pred)
{
    typename container_type::accessor acc;

    if (m_data_->find(acc, s))
    {
        for (auto const &p:acc->second)
        {
            acc->second.remove_if([&](value_type const &p) { return pred(p, s); });
        }
    }
}


template<typename V, typename K> template<typename Predicate> void
Particle<V, K>::remove_if(range_type const &r0, Predicate const &pred)
{
    parallel::parallel_for(r0, [&](range_type const &r) { for (auto const &s:r) { remove_if(pred, s); }});
}


}
} //namespace simpla { namespace particle


#endif //SIMPLA_PARTICLEV000_H
