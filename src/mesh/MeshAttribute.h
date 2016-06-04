/**
 * @file MeshAttribute.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESHATTRIBUTE_H
#define SIMPLA_MESHATTRIBUTE_H

#include "../base/Object.h"
#include "../gtl/Log.h"
#include "../gtl/MemoryPool.h"
#include "../data_model/DataSet.h"
#include "Mesh.h"
#include "MeshBase.h"
#include "MeshEntity.h"


namespace simpla { namespace mesh
{

/**
 *  PlaceHolder class of MeshAttribute
 */
struct MeshAttribute : public base::Object, public Acceptor
{
    SP_OBJECT_HEAD(MeshAttribute, base::Object)

public:

    MeshAttribute() { }

    virtual ~MeshAttribute() { }

    MeshAttribute(MeshAttribute const &other) = delete;

    MeshAttribute(MeshAttribute &&other) = delete;

    MeshAttribute &operator=(MeshAttribute const &) = delete;

    void swap(MeshAttribute &other) = delete;

    struct View
    {
        View() { }

        View(View const &other) { }

        View(View &&other) { }

        ~View() { };

        View &operator=(View const &other) = delete;

        virtual bool is_a(std::type_info const &t_info) const = 0;

        template<typename T> inline bool is_a() const { return (std::is_base_of<View, T>::value && is_a(typeid(T))); }

        virtual std::string get_class_name() const = 0;

        virtual void swap(View &other) = 0;

        virtual bool is_valid() const = 0;

        virtual bool deploy() = 0;

        virtual void clear() = 0;

        virtual MeshEntityType entity_type() const = 0;

        virtual MeshEntityRange const &range() const = 0;

        virtual void data_set(data_model::DataSet const &) = 0;

        virtual void data_set(mesh::MeshEntityRange const &, data_model::DataSet &) = 0;

        virtual data_model::DataSet data_set() const = 0;

        virtual data_model::DataSet data_set(mesh::MeshEntityRange const &) const = 0;
    };

    /** register MeshBlockId to attribute data collection.  */

    template<typename TF, typename ...Args>
    std::shared_ptr<TF> add(MeshBase const *m, Args &&...args)
    {
        assert(m != nullptr);

        std::shared_ptr<TF> res;

        static_assert(std::is_base_of<View, TF>::value,
                      "Object is not a mesh::MeshAttribute::View");
        auto it = m_attrs_.find(m->uuid());

        if (it != m_attrs_.end())
        {

            if (!it->second->template is_a<TF>())
            {
                RUNTIME_ERROR << "Attribute type cast error! "
                << "From:" << it->second->get_class_name()
                << " To: " << traits::type_id<typename TF::mesh_type>::name() <<
                std::endl;
            }

            res = std::make_shared<TF>(*std::dynamic_pointer_cast<TF>(it->second));
        }
        else
        {
            if (!m->template is_a<typename TF::mesh_type>())
            {
                RUNTIME_ERROR << "Mesh type cast error! "
                << "From:" << m->get_class_name()
                << " To: " << traits::type_id<typename TF::mesh_type>::name() <<
                std::endl;
            }
            else
            {
                res = std::make_shared<TF>(m, std::forward<Args>(args)...);

                m_attrs_.emplace(std::make_pair(m->uuid(), std::dynamic_pointer_cast<View>(res)));


            }
        }

        return res;

    }

    std::shared_ptr<View> get(MeshBlockId const &id)
    {
        std::shared_ptr<View> res(nullptr);
        auto it = m_attrs_.find(id);
        if (it != m_attrs_.end()) { res = it->second; }
        return res;
    }

    std::shared_ptr<const View> get(MeshBlockId const &id) const
    {
        std::shared_ptr<View> res(nullptr);
        auto it = m_attrs_.find(id);
        if (it != m_attrs_.end()) { res = it->second; }
        return res;
    }

    /** erase MeshBlockId from attribute data collection.  */
    size_t erase(MeshBlockId const &id)
    {
        return m_attrs_.erase(id);
    }

    data_model::DataSet dataset(MeshBlockId const &id) const
    {
        return m_attrs_.at(id)->data_set();
    }

    void dataset(MeshBlockId const &id, data_model::DataSet const &d)
    {
        try
        {
            return m_attrs_.at(id)->data_set(d);

        }
        catch (std::out_of_range const &)
        {
            RUNTIME_ERROR << "Block [" << boost::uuids::hash_value(id) << "] is missing!" << std::endl;
        }
    }

    void dataset(std::map<MeshBlockId, data_model::DataSet> *res) const
    {
        for (auto const &item:m_attrs_)
        {
            res->emplace(std::make_pair(item.first, item.second->data_set()));
        };
    }

    void dataset(std::map<MeshBlockId, data_model::DataSet> const &d)
    {
        for (auto const &item:d) { dataset(item.first, item.second); }
    }

    bool has(MeshBlockId const &id) const { return m_attrs_.find(id) != m_attrs_.end(); }

protected:
    std::map<MeshBlockId, std::shared_ptr<View>> m_attrs_;
};


}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESHATTRIBUTE_H
