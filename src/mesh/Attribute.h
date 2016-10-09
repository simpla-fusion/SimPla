/**
 * @file MeshAttribute.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESHATTRIBUTE_H
#define SIMPLA_MESHATTRIBUTE_H

#include "../toolbox/Object.h"
#include "../toolbox/Log.h"
#include "../toolbox/MemoryPool.h"
#include "../toolbox/DataSet.h"
#include "MeshCommon.h"
#include "Chart.h"
#include "EntityRange.h"


namespace simpla { namespace mesh
{
struct Attribute
{

    Attribute() {};

    virtual ~Attribute() {};

    Attribute &operator=(Attribute const &other) = delete;

    virtual void deploy()=0;

    virtual bool is_deployed() const { return true; };

    virtual void clear() = 0;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual bool is_a(std::type_info const &t_info) const = 0;

    template<typename T>
    inline bool is_a() const { return (std::is_base_of<Attribute, T>::value && is_a(typeid(T))); }

    virtual bool is_valid() const = 0;

    virtual bool empty() const = 0;

    virtual std::string get_class_name() const = 0;

    virtual EntityRange entity_id_range(MeshEntityStatus status = SP_ES_ALL) const = 0;

    virtual MeshEntityType entity_type() const = 0;

    virtual size_type entity_size_in_byte() const = 0;

    virtual size_type size_in_byte() const = 0;

    virtual Chart const *mesh() const = 0;

    virtual std::shared_ptr<void> data() = 0;

    virtual std::shared_ptr<const void> data() const = 0;


//    virtual void dataset(data_model::DataSet const &) = 0;
//    virtual void dataset(mesh::EntityRange const &, data_model::DataSet const &) = 0;
};
//
///**
// *  PlaceHolder class of Attribute
// */
//struct Attribute : public toolbox::Object, public Acceptor
//{
//    SP_OBJECT_HEAD(Attribute, toolbox::Object)
//
//public:
//
//    Attribute() { }
//
//    virtual Attribute }
//
//    Attribute(Attribute const &other) = delete;
//
//    Attribute(Attribute &&other) = delete;
//
//    Attribute &operator=(Attribute const &) = delete;
//
//    void swap(Attribute &other) = delete;
//
//    virtual std::ostream &print(std::ostream &os, int indent = 1) const
//    {
//        for (auto const &item:m_views_)
//        {
////            os << std::setw(indent + 1) << " id=" << boost::uuids::hash_value(item.first) << ",";
//            item.second->print(os, indent + 2);
////            os << "";
//
//        }
//        return os;
//    }
//
//    /** register MeshBlockId to attribute m_data collection.  */
//
//    template<typename TF, typename ...Args>
//    std::shared_ptr<TF> add(Chart const *m, Args &&...args)
//    {
//        assert(m != nullptr);
//
//        std::shared_ptr<TF> res;
//
//        static_assert(std::is_base_of<View, TF>::entity,
//                      "Object is not a get_mesh::Attribute::View");
//        auto it = m_views_.find(m->uuid());
//
//        if (it != m_views_.end())
//        {
//
//            if (!it->second->template is_a<TF>())
//            {
//                RUNTIME_ERROR << "Attribute type cast error! "
//                << "From:" << it->second->get_class_name()
//                << " To: " << traits::type_id<typename TF::mesh_type>::name() <<
//                std::endl;
//            }
//
//            res = std::make_shared<TF>(*std::dynamic_pointer_cast<TF>(it->second));
//        }
//        else
//        {
//            if (!m->template is_a<typename TF::mesh_type>())
//            {
//                RUNTIME_ERROR << "Mesh type cast error! "
//                << "From:" << m->get_class_name()
//                << " To: " << traits::type_id<typename TF::mesh_type>::name() <<
//                std::endl;
//            }
//            else
//            {
//                res = std::make_shared<TF>(m, std::forward<Args>(args)...);
//
//                m_views_.emplace(std::make_pair(m->uuid(), std::dynamic_pointer_cast<View>(res)));
//
//
//            }
//        }
//
//        return res;
//
//    }
//
//    std::shared_ptr<View> get(MeshBlockId const &id)
//    {
//        std::shared_ptr<View> res(nullptr);
//        auto it = m_views_.find(id);
//        if (it != m_views_.end()) { res = it->second; }
//        return res;
//    }
//
//    std::shared_ptr<const View> get(MeshBlockId const &id) const
//    {
//        std::shared_ptr<View> res(nullptr);
//        auto it = m_views_.find(id);
//        if (it != m_views_.end()) { res = it->second; }
//        return res;
//    }
//
//    /** erase MeshBlockId from attribute m_data collection.  */
//    size_t erase(MeshBlockId const &id)
//    {
//        return m_views_.erase(id);
//    }
//
//    data_model::DataSet dataset(MeshBlockId const &id) const
//    {
//        return m_views_.at(id)->dataset();
//    }
//
//    void dataset(MeshBlockId const &id, data_model::DataSet const &d)
//    {
//        try
//        {
//            return m_views_.at(id)->dataset(d);
//
//        }
//        catch (std::out_of_range const &)
//        {
//            RUNTIME_ERROR << "Block [" << boost::uuids::hash_value(id) << "] is missing!" << std::endl;
//        }
//    }
//
//    void dataset(std::map<MeshBlockId, data_model::DataSet> *res) const
//    {
//        for (auto const &item:m_views_)
//        {
//            res->emplace(std::make_pair(item.first, item.second->dataset()));
//        };
//    }
//
//    void dataset(std::map<MeshBlockId, data_model::DataSet> const &d)
//    {
//        for (auto const &item:d) { dataset(item.first, item.second); }
//    }
//
//    bool has(MeshBlockId const &id) const { return m_views_.find(id) != m_views_.end(); }
//
//protected:
//    std::map<MeshBlockId, std::shared_ptr<View>> m_views_;
//};


}}//namespace simpla{namespace get_mesh{

#endif //SIMPLA_MESHATTRIBUTE_H
