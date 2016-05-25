/**
 * @file context.h
 *
 * @date    2014-9-18  AM9:33:53
 * @author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <memory>
#include <list>
#include <map>
#include <boost/uuid/uuid.hpp>
#include "../gtl/primitives.h"
#include "../base/Object.h"
#include "../mesh/MeshEntity.h"

namespace simpla { namespace mesh { struct MeshBase; }}//namespace simpla { namespace mesh {


namespace simpla { namespace task_flow
{


class Attribute;

class Worker;

class Context
{
    typedef Context this_type;
public:


    void apply(Worker &w, uuid const &id, Real dt);

    void sync(Worker &w, uuid const &oid);


private:


    std::map<std::string, std::weak_ptr<Attribute>> m_attributes_;

public:

    template<typename TF>
    std::shared_ptr<TF> get_attribute(std::string const &s_name)
    {
        static_assert(std::is_base_of<Attribute, TF>::value, "TF is not a Attribute");

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
        static_assert(std::is_base_of<Attribute, TF>::value, "TF is not a Attribute");

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
        static_assert(std::is_base_of<Attribute, TF>::value, "TF is not a Attribute");

        m_attributes_.insert(std::make_pair(name, std::dynamic_pointer_cast<Attribute>(p)));
    };


};

struct Attribute : public base::Object
{
    SP_OBJECT_HEAD(Attribute, base::Object);


    std::map<uuid, std::shared_ptr<void>> m_data_tree_;

    uuid m_id_;

    Attribute(mesh::MeshAtlas const &mesh_tree, std::string const &name)
            : m_mesh_tree_(mesh_tree)
    {
    }

    virtual void view(uuid const &id)
    {
        assert(m_data_tree_.find(id) != m_data_tree_.end());
        m_id_ = id;

    };

    void *data() { return m_data_tree_.at(m_id_).get(); }

    void const *data() const { return m_data_tree_.at(m_id_).get(); }

    mesh::Mesh const *mesh() const { return m_mesh_tree_.at(m_id_); }


};

struct Worker
{
    typedef Context context_type;

    std::list<std::string> m_attributes_;

    context_type &m_ctx_;

    Worker(context_type &ctx) : m_ctx_(ctx) { }

    virtual ~Worker() { }

    void view(uuid const &id)
    {
        for (auto &item:m_attributes_)
        {
            m_ctx_.m_attributes_.at(item).lock()->view(id);
        }
    }

    virtual void work(Real dt) = 0;

    /**
    * copy data from lower level
    */
    virtual void coarsen(uuid const &) { }

    /**
     * copy data to lower level
     */
    virtual void refine(uuid const &) { }

    /**
     * copy data from same level neighbour
     */
    virtual void sync(std::list<uuid> const &) { }
};


}}// namespace simpla//namespace task_flow

#endif /* CORE_APPLICATION_CONTEXT_H_ */
