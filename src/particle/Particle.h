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
#include "../gtl/design_pattern/SingletonHolder.h"
#include "../gtl/utilities/MemoryPool.h"
#include "../base/DataObject.h"

namespace simpla { template<typename ...> class Field; }
namespace simpla { namespace particle
{

struct ParticleBase : public base::DataObject
{
private:
    typedef ParticleBase this_type;
public:
    SP_OBJECT_HEAD(ParticleBase, base::DataObject);

    ParticleBase() { }

    virtual ~ParticleBase() { }

    virtual std::ostream &print(std::ostream &os, int indent) const = 0;

    virtual Properties const &properties() const = 0;

    virtual Properties &properties() = 0;

    virtual data_model::DataSet data_set() const = 0;

    virtual size_t size() const = 0;

    virtual void deploy() = 0;

    virtual void sync() = 0;

    virtual void rehash() = 0;

    virtual void push(Real t0, Real t1) = 0;

    virtual void integral() = 0;

    virtual void apply_filter() = 0;

    std::ostream &operator<<(std::ostream &os) const { return this->print(os, 0); }

};

inline std::ostream &operator<<(std::ostream &os, ParticleBase const &p) { return p.print(os, 0); };

template<typename ...> struct ParticleContainer;
template<typename ...> struct Particle;

template<typename P, typename M>
struct Particle<P, M> : public ParticleBase
{
private:
    typedef ParticleContainer<P, M> container_type;
    typedef Particle<P, M> this_type;
    std::shared_ptr<container_type> m_data_;
public:

    typedef M mesh_type;
    typedef P engine_type;
    typedef typename engine_type::sample_type sample_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::range_type range_type;
    static constexpr int iform = container_type::iform;

    Particle(mesh_type &m, std::string const &s_name) : m_data_(new container_type(m, s_name))
    {
        if (s_name != "") { m.enroll(s_name, m_data_); }
    }

    ~Particle() { }

    Particle(this_type const &other) : m_data_(other.m_data_) { };

    Particle(this_type &&other) : m_data_(other.m_data_) { };

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type const &other) { std::swap(other.m_data_, m_data_); }

    mesh_type const &mesh() const { return m_data_->mesh(); }

    engine_type const &engine() const { return *std::dynamic_pointer_cast<const engine_type>(m_data_); }

    engine_type &engine() { return *std::dynamic_pointer_cast<engine_type>(m_data_); }
//
//    std::shared_ptr<container_type> data() { return m_data_; }
//
//    std::shared_ptr<container_type> const data() const { return m_data_; }
//
//    std::shared_ptr<typename mesh_type::AttributeEntity> attribute()
//    {
//        return std::dynamic_pointer_cast<typename mesh_type::AttributeEntity>(m_data_);
//    }
//
//    std::shared_ptr<typename mesh_type::AttributeEntity> const attribute() const
//    {
//        return std::dynamic_pointer_cast<typename mesh_type::AttributeEntity>(m_data_);
//    }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        return m_data_->print(os, indent);
    }

    virtual data_model::DataSet data_set() const { return m_data_->data_set(); }

    virtual Properties const &properties() const { return m_data_->properties(); };

    virtual Properties &properties() { return m_data_->properties(); };

    virtual size_t size() const { return m_data_->count(); }

    virtual void deploy() { m_data_->deploy(); }

    virtual void clear() { m_data_->clear(); }

    virtual void sync() {/* m_data_->sync();*/ }

    virtual void rehash() { /*m_data_->rehash();*/ }

    virtual void push(Real t0, Real t1)
    {
        CMD << "Push particle " << m_data_->name() << std::endl;

        m_data_->filter([&](sample_type *p) { m_data_->engine_type::push(t0, t1, p); });
    }

    virtual void integral() { for (auto &item:m_integral_list_) { item.second(item.first); }}


    virtual void add_filter(std::string const &key,
                            std::function<void(typename container_type::bucket_type &, id_type)> const &f,
                            range_type const &r)
    {
        m_filter_list_.push_back(std::make_tuple(key, f, r));
    }

    virtual void apply_filter()
    {
        for (auto const &item:m_filter_list_)
        {
            VERBOSE << " Apply filter [" << std::get<0>(item) << "] to " << m_data_->name() << std::endl;

            m_data_->foreach_bucket(std::get<1>(item), std::get<2>(item));
        }
    }

    template<typename TGen> void generate(TGen &gen, id_type s);

    template<typename TGen, typename TRange> void generate(TGen &gen, TRange const &);

    template<typename TGen> void generate(TGen &);



    //! @name as particle
    //! @{

    template<typename TV, int IFORM, typename ...Policies>
    void add_gather(Field<TV, mesh_type, std::integral_constant<int, IFORM>, Policies...> &f);


    template<typename TField, typename TRange>
    void gather(TField *res, TRange const &r) const;

    template<typename TField>
    void gather(TField *res) const;

    //! @}
private:
    std::list<std::tuple<std::string, std::function<void(typename container_type::bucket_type &, id_type)>,
            typename mesh_type::range_type  >> m_filter_list_;


    std::list<std::pair<std::weak_ptr<typename mesh_type::AttributeEntity>, std::function<void(
            std::weak_ptr<typename mesh_type::AttributeEntity> &)> >> m_integral_list_;
};
//**************************************************************************************************

template<typename P, typename M>
template<typename TGen> void
Particle<P, M>::generate(TGen &gen, id_type s)
{
    auto g = gen(s);

    m_data_->insert(std::get<0>(g), std::get<1>(g), s);
}


template<typename P, typename M>
template<typename TGen, typename TRange> void
Particle<P, M>::generate(TGen &gen, const TRange &r0)
{
    parallel::parallel_for(r0, [&](TRange const &r) { for (auto const &s:r) { generate(gen, s); }});
}

template<typename P, typename M>
template<typename TGen> void
Particle<P, M>::generate(TGen &gen)
{
    generate(gen, m_data_->mesh().template range<container_type::iform>());
//    m_data_->mesh().template for_each_boundary<container_type::iform>(
//            [&](range_type const &r) { generate(gen, r); });
//
//    parallel::DistributedObject dist_obj;
//
//    m_data_->sync_(*m_data_, &dist_obj, false);
//
//    dist_obj.sync();
//
//    m_data_->mesh().template for_each_center<container_type::iform>(
//            [&](range_type const &r) { generate(gen, r); });
//
//    dist_obj.wait();
//
//    for (auto const &item :  dist_obj.recv_buffer)
//    {
//        sample_type const *p = reinterpret_cast<sample_type const *>(std::get<1>(item).data.get());
//        m_data_->insert(p, p + std::get<1>(item).memory_space.size());
//    }
}


//*******************************************************************************



template<typename P, typename M> template<typename TField, typename TRange> void
Particle<P, M>::gather(TField *J, TRange const &r0) const
{
    parallel::parallel_for(r0, [&](TRange const &r)
    {
        for (auto const &s:r)
        {
            typename TField::field_value_type tmp;

            tmp = 0;

            id_type neighbours[mesh_type::MAX_NUM_OF_NEIGHBOURS];

            int num = m_data_->mesh().get_adjacent_cells(container_type::iform, s, neighbours);
            auto x0 = m_data_->mesh().point(s);

            for (int i = 0; i < num; ++i)
            {
                typename container_type::const_accessor acc1;

                if (m_data_->find(acc1, neighbours[i]))
                {
                    for (auto const &p:acc1->second) { m_data_->engine_type::integral(x0, p, &tmp); }
                }
            }

            J->assign(s, tmp);
        }
    });
};

template<typename P, typename M> template<typename TField> void
Particle<P, M>::gather(TField *J) const
{

    typedef typename mesh_type::range_type range_t;

    static constexpr int f_iform = traits::iform<TField>::value;

    J->mesh().template for_each_boundary<f_iform>([&](range_t const &r) { gather(J, r); });

    parallel::DistributedObject dist_obj;
    dist_obj.add(*J);
    dist_obj.sync();

    J->mesh().template for_each_center<f_iform>([&](range_t const &r) { gather(J, r); });

    dist_obj.wait();


}

//**************************************************************************************************

template<typename P, typename M>
template<typename TV, int IFORM, typename ...Policies>
void  Particle<P, M>::add_gather(::simpla::Field<TV, M, std::integral_constant<int, IFORM>, Policies...> &f)
{
    m_integral_list_.push_back(
            std::make_pair(
                    std::weak_ptr<typename M::AttributeEntity>(
                            std::dynamic_pointer_cast<typename M::AttributeEntity>(f.data())),
                    [&](std::weak_ptr<typename M::AttributeEntity> &f_entity)
                    {
                        auto fp = f_entity.lock();

                        ASSERT(fp->is_a(typeid(typename mesh_type::template Attribute<TV, IFORM>)));

                        ::simpla::Field<TV, M, std::integral_constant<int, IFORM>, Policies...> f{
                                std::dynamic_pointer_cast<typename M::template Attribute<TV, IFORM>>(
                                        fp)};

                        CMD << "Integral particle [" << m_data_->name()
                        << "] to Field [" << fp->name() << "]" << std::endl;

                        gather(&f);

                    }));
}
}} //namespace simpla


#endif /* CORE_PARTICLE_PARTICLE_H_ */
