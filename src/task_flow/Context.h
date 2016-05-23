/**
 * @file context.h
 *
 * @date    2014-9-18  AM9:33:53
 * @author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <list>
#include "../mesh/MeshAtlas.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

namespace simpla { namespace mesh
{

class MeshBase;

class MeshAtlas;

}}//namespace simpla{ namespace mesh{

namespace simpla { namespace task_flow
{
typedef boost::uuids::uuid uuid;


class Context
{
    typedef Context this_type;
public:

    class Attribute;

    class Updater;

    class Mapper;

    uuid m_root_;


    void update(Real dt) { update(m_root_, dt); };

    void update(uuid id, Real dt);

private:
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> Graph;

    Graph m_mesh_atlas_;

    std::map<uuid, std::map<uuid, std::shared_ptr<Attribute>>> m_attributes_;

public:


//    void next_time_step() { m_time_ += m_dt_; }
//
//    double time() const { return m_time_; }
//
//    void time(double t) { m_time_ = t; }
//
//    double dt() const { return m_dt_; }
//
//    void dt(double p_dt) { m_dt_ = p_dt; }
    std::shared_ptr<mesh::MeshBase> mesh(uuid id) const
    {

    }

    template<typename TF>
    std::shared_ptr<TF> get_attribute(std::string const &s_name)
    {
        static_assert(std::is_base_of<Attribute, TF>::value);

        auto it = m_attributes_.find(s_name);

        if (it == m_attributes_.end())
        {
            return create_attribute<TF>(s_name);
        }
        else if (it->second.lock()->is_a(typeid(TF)))
        {
            return std::dynamic_pointer_cast<TF>(it->second.lock());
        }
        else
        {
            return nullptr;
        }

    }

    template<typename TF>
    std::shared_ptr<TF> create_attribute(std::string const &s_name = "")
    {
        static_assert(std::is_base_of<Attribute, TF>::value);

        auto res = std::make_shared<TF>(*this);

        if (s_name != "") { enroll(s_name, std::dynamic_pointer_cast<Attribute>(res)); }

        return res;
    }

    template<typename TF>
    std::shared_ptr<TF> create_attribute() const
    {
        return std::make_shared<TF>(*this);
    }

    template<typename TF>
    void enroll(std::string const &name, std::shared_ptr<TF> p)
    {
        static_assert(std::is_base_of<Attribute, TF>::value);

        m_attributes_.insert(std::make_pair(name, std::dynamic_pointer_cast<Attribute>(p)));
    };


};

struct Context::Worker
{

    Context &m_ctx_;

    std::list<std::shared_ptr<Attribute> > m_attr_id_;

    Worker() { }

    Worker(Context &ctx) : m_ctx_(ctx) { }

    virtual ~Worker() { }

    void view(mesh::uuid const &o_mesh_id)
    {
        for (auto &attr:m_attr_id_)
        {
            attr->swap(m_ctx_.attr_map[attr->id()][o_mesh_id]);

        }
    }

    virtual void work(Real dt) = 0;

    /**
    * copy data from lower level
    */
    virtual void coarsen(mesh::uuid const &) { }

    /**
     * copy data to lower level
     */
    virtual void refine(mesh::uuid const &) { }

    /**
     * copy data from same level neighbour
     */
    virtual void sync(std::list<mesh::uuid> const &) { }
};

struct Context::Attribute :
        public base::Object,
        public std::enable_shared_from_this<Attribute>
{
    SP_OBJECT_HEAD(Attribute, base::Object);

    uuid m_attr_id_;

    mesh::uuid m_mesh_id_;

public:

    Attribute(Context const &ctx) : m_ctx_(ctx) { }

    virtual ~Attribute() { }

    virtual void view(mesh::uuid const &id) { m_id_ = id; };

    mesh::uuid id() const { return m_id_; }

    template<typename T>
    std::shared_ptr<T> data()
    {
        assert(m_data_tree_.find(m_id_) != m_data_tree_.end());
        return m_data_tree_.at(m_id_).template as<T>();
    }

    template<typename T>
    T const *mesh() const
    {
        return std::dynamic_pointer_cast<T>(m_ctx_.mesh(m_id_)).get();
    }

    virtual void swap(Attribute &other)
    {
        std::swap(m_mesh_id_, other.m_mesh_id_);
    };
};


}//namespace task_flow
}// namespace simpla

#endif /* CORE_APPLICATION_CONTEXT_H_ */
