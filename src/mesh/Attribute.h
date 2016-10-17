/**
 * @file MeshAttribute.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESHATTRIBUTE_H
#define SIMPLA_MESHATTRIBUTE_H

#include "../toolbox/Object.h"
#include "../toolbox/Log.h"
#include "../toolbox/Memory.h"
#include "../toolbox/DataSet.h"
#include "MeshCommon.h"
#include "MeshBase.h"


namespace simpla { namespace mesh
{
struct AttributeBase
{
    AttributeBase() {};

    virtual ~AttributeBase() {};

    AttributeBase &operator=(AttributeBase const &other) = delete;

    virtual void deploy() = 0;

    virtual void clear() = 0;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        UNIMPLEMENTED;
        return os;
    }

    virtual bool is_a(std::type_info const &t_info) const = 0;

    template<typename T> inline bool is_a() const
    {
        return (std::is_base_of<AttributeBase, T>::value && is_a(typeid(T)));
    }

    virtual bool is_valid() const = 0;

    virtual bool empty() const = 0;

    virtual std::string get_class_name() const = 0;

    virtual MeshBase const *mesh() const = 0;

    virtual void *raw_data() = 0;

    virtual void const *raw_data() const = 0;

    virtual void dataset(toolbox::DataSet const &) = 0;

    virtual toolbox::DataSet dataset() const =0;
};

template<typename V, typename M, MeshEntityType IFORM = VERTEX>
class Attribute : public AttributeBase
{


public:
    typedef Attribute<V, M, IFORM> this_type;
    static constexpr MeshEntityType iform = IFORM;
    typedef M mesh_type;
    typedef V value_type;

    Attribute(mesh_type const *m, std::shared_ptr<value_type> p = nullptr) : m_mesh_(m), m_data_(p) {};

    Attribute(Attribute const &other) : m_mesh_(other.m_mesh_), m_data_(other.m_data_) {};

    virtual ~Attribute() {};

    Attribute &operator=(Attribute const &other)
    {
        this_type(other).swap(*this);
        return *this;
    };

    void swap(this_type &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
        std::swap(m_data_, other.m_data_);
    }


    virtual void deploy()
    {
        assert(m_mesh_ != nullptr);
        if (!empty()) { m_data_ = toolbox::MemoryHostAllocT<value_type>(m_mesh_->number_of_entities(IFORM)); }
    };

    virtual void clear()
    {
        deploy();
        toolbox::MemorySet(m_data_, 0, m_mesh_->number_of_entities(IFORM) * sizeof(value_type));
    }


    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }

    virtual bool is_valid() const { return m_mesh_ != nullptr && !empty(); };

    virtual bool empty() const { return m_data_ != nullptr; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name()
    {
        return std::string("Attribute<") +
               traits::type_id<value_type>::name() + "," +
               traits::type_id<mesh_type>::name() + "," +
               traits::type_id<index_const<IFORM>>::name()
               + ">";
    }

    virtual mesh_type const *mesh() const { return m_mesh_; };

    virtual void *raw_data() { return m_data_.get(); }

    virtual void const *raw_data() const { return m_data_.get(); }

    virtual std::shared_ptr<value_type> data() { return m_data_; }

    virtual std::shared_ptr<const value_type> data() const { return m_data_; }

    virtual void dataset(toolbox::DataSet const &) { UNIMPLEMENTED; };

    virtual toolbox::DataSet dataset() const
    {
        toolbox::DataSet res;

        res.data_type = toolbox::DataType::create<value_type>();

        res.data = m_data_;

        std::tie(res.memory_space, res.data_space) = m_mesh_->data_space(IFORM);

        return res;
    };

    inline value_type &get(mesh::MeshEntityId const &s) { return (m_data_.get())[m_mesh_->hash(s)]; }

    inline value_type const &get(mesh::MeshEntityId const &s) const { return (m_data_.get())[m_mesh_->hash(s)]; }

    inline value_type &operator[](mesh::MeshEntityId const &s) { return get(s); }

    inline value_type const &operator[](mesh::MeshEntityId const &s) const { return get(s); }

    template<typename TOP> void
    apply(TOP const &op, EntityRange const &r0, value_type const &v)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), v); });
    }

    template<typename TOP, typename TOther> void
    apply(TOP const &op, EntityRange const &r0, TOther const &fun)
    {
        deploy();
        r0.foreach([&](MeshEntityId const &s) { op(get(s), fun(s)); });
    }

    template<typename TOP> void
    apply(TOP const &op, value_type const &v)
    {
        deploy();
        m_mesh_->foreach(iform, [&](mesh::MeshEntityId const &s) { op(get(s), v); });
    }

    template<typename TOP, typename TFun> void
    apply(TOP const &op, TFun const &fun)
    {
        deploy();
        m_mesh_->foreach(iform, [&](MeshEntityId const &s) { op(get(s), fun(s)); });
    }


protected:
    M const *m_mesh_;
    std::shared_ptr<V> m_data_;
};

//
//template<typename ...U, typename TFun> Field<U...> &
//assign(Field<U...> &f,mesh::EntityRange const &r0,  TFun const &op,
//      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::value_type, TFun(mesh::MeshEntityId const &)))
//{
//    f.deploy();
//
//    auto const &m = *f.mesh();
//
//    static const mesh::MeshEntityType IFORM = Field<U...>::iform;
//
//    r0.foreach([&](mesh::MeshEntityId const &s) { f[s] = op(s); });
//
//    return f;
//};
//
//template<typename ...U, typename TFun> Field<U...> &
//assign(Field<U...> &f, mesh::EntityRange const &r0, TFun const &op,
//      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::value_type, TFun(typename Field<U...>::value_type & )))
//{
//    f.deploy();
//
//    auto const &m = *f.mesh();
//
//    static const mesh::MeshEntityType IFORM = Field<U...>::iform;
//
//    r0.foreach([&](mesh::MeshEntityId const &s) { op(f[s]); });
//
//    return f;
//}
//
//template<typename ...U, typename ...V> Field<U...> &
//assign(Field<U...> &f,mesh::EntityRange const &r0,  Field<V...> const &g)
//{
//    f.deploy();
//
//    auto const &m = *f.mesh();
//
//    static const mesh::MeshEntityType IFORM = Field<U...>::iform;
//
//    r0.foreach([&](mesh::MeshEntityId const &s) { f[s] = g[s]; });
//
//    return f;
//}


//template<typename V, typename M, MeshEntityType IFORM> constexpr MeshEntityType Attribute<V, M, IFORM>::iform = IFORM;
//
//**
// *  PlaceHolder class of AttributeBase
// */
//struct AttributeBase : public toolbox::Object, public Acceptor
//{
//    SP_OBJECT_HEAD(AttributeBase, toolbox::Object)
//
//public:
//
//    AttributeBase() { }
//
//    virtual AttributeBase }
//
//    AttributeBase(AttributeBase const &other) = delete;
//
//    AttributeBase(AttributeBase &&other) = delete;
//
//    AttributeBase &operator=(AttributeBase const &) = delete;
//
//    void swap(AttributeBase &other) = delete;
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
//    std::shared_ptr<TF> add(MeshBase const *m, Args &&...args)
//    {
//        assert(m != nullptr);
//
//        std::shared_ptr<TF> res;
//
//        static_assert(std::is_base_of<View, TF>::entity,
//                      "Object is not a get_mesh::AttributeBase::View");
//        auto it = m_views_.find(m->uuid());
//
//        if (it != m_views_.end())
//        {
//
//            if (!it->second->template is_a<TF>())
//            {
//                RUNTIME_ERROR << "AttributeBase type cast error! "
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
//            RUNTIME_ERROR << "MeshBase [" << boost::uuids::hash_value(id) << "] is missing!" << std::endl;
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
