/**
 * @file Particle.h
 *
 *  created on: 2012-11-1
 *      Author: salmon
 */

#ifndef CORE_PARTICLE_PARTICLE_H_
#define CORE_PARTICLE_PARTICLE_H_

#include <memory>
#include "../toolbox/type_traits.h"
#include "../toolbox/Properties.h"

namespace simpla { template<typename ...> struct Field; }

namespace simpla { namespace particle
{
/**  @addtogroup physics Physics */

template<typename ...> struct Particle;


template<typename P, typename M, typename ...Others>
struct Particle<P, M, Others...>
        : public P,
          public Field<spPage *, M, simpla::int_const<mesh::VOLUME> >,
          public std::enable_shared_from_this<Particle<P, M, Others...>>
{
private:

    typedef Particle<P, M, Others...> this_type;
    typedef mesh::MeshAttribute::View View;
    typedef mesh::MeshAttribute::View base_type;
public:

    typedef M mesh_type;
    typedef P engine_type;
    typedef Field<spPage *, M, simpla::int_const<mesh::VOLUME> > field_type;
    typedef typename P::point_s value_type;
    typedef typename mesh::MeshEntityId id_type;
    typedef typename mesh::MeshEntityRange range_type;

    using typename P::point_s;

private:
    std::shared_ptr<struct spPagePool> m_pool_;

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


    template<typename TFactory, typename ... Args, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
    Particle(TFactory &factory, Args &&...args)
            : m_properties_(nullptr), m_pool_(nullptr)
    {
        field_type::m_pimpl_ = (std::dynamic_pointer_cast<base_type>(
                factory.template create<this_type>(std::forward<Args>(args)...)));
        deploy();
    }

    //CreateNew construct
    Particle(this_type const &other)
            : engine_type(other), field_type(other), m_properties_(other.m_properties_), m_pool_(other.m_pool_)
    {
    }


    // Move construct
    Particle(this_type &&other)
            : engine_type(other), field_type(other), m_properties_(other.m_properties_), m_pool_(other.m_pool_)
    {
    }

    virtual ~Particle() { }

    Particle<P, M, Others...> &operator=(Particle<P, M, Others...> const &other)
    {
        Particle<P, M, Others...>(other).swap(*this);
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
    using field_type::get;

    std::ostream &print(std::ostream &os, int indent) const;

    virtual mesh_type const &mesh() const { return *this->m_mesh_; }

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || t_info == typeid(field_type);
    };

    virtual std::string getClassName() const { return class_name(); };

    static std::string class_name() { return "Particle<" + traits::type_id<P, M>::name() + ">"; };

    virtual bool deploy();

    virtual data_model::DataSet dataset() const;

    virtual data_model::DataSet dataset(mesh::MeshEntityRange const &) const;

    virtual void dataset(data_model::DataSet const &);

    virtual void dataset(mesh::MeshEntityRange const &, data_model::DataSet const &);

    virtual size_t size() const { return count(entity_id_range()); }

    virtual void clear();

    //**************************************************************************************************
    // as particle

    template<typename TV, typename ...Others, typename ...Args>
    void gather(Field<TV, mesh_type, Others...> *res, Args &&...args) const;

    template<typename ...Args> void update(Args &&...args);


    //**************************************************************************************************
    //! @name as container
    //! @{
    void insert(id_type const &s, point_s p)
    {
        spInsert(get(s), 1, engine_type::ele_size_in_byte(), &p);
    };

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
        apply(entity_id_range(), op, std::forward<Args>(args)...);
    };

    template<typename TOP, typename ...Args>
    void apply(TOP const &op, Args &&...args) const
    {
        apply(entity_id_range(), op, std::forward<Args>(args)...);
    };


};//class Particle

}}//namespace simpla { namespace particle

#endif /* CORE_PARTICLE_PARTICLE_H_ */
