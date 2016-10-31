/**
 * @file MeshAttribute.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESHATTRIBUTE_H
#define SIMPLA_MESHATTRIBUTE_H


namespace simpla { namespace data
{
class DataBase;

struct PatchBase
{
    PatchBase() {};

    virtual ~PatchBase() {};

    PatchBase &operator=(PatchBase const &other) = delete;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const =0;

    virtual void load(DataBase const &, std::string const & = "") {};

    virtual void save(DataBase *, std::string const & = "") const {};

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(PatchBase); }

    virtual void deploy() = 0;

    virtual void clear() = 0;

    virtual mesh::MeshEntityType entity_type() const =0;

    virtual bool is_valid() const = 0;

    virtual bool empty() const = 0;

    virtual std::string get_class_name() const = 0;

    virtual void *data() = 0;

    virtual void const *data() const = 0;
};


//
//template<typename ...U, typename TFun> Field<U...> &
//assign(Field<U...> &f,mesh_as::EntityRange const &r0,  TFun const &op,
//      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::value_type, TFun(mesh_as::MeshEntityId const &)))
//{
//    f.deploy();
//
//    auto const &m = *f.mesh_as();
//
//    static const mesh_as::MeshEntityType IFORM = Field<U...>::iform;
//
//    r0.foreach([&](mesh_as::MeshEntityId const &s) { f[s] = op(s); });
//
//    return f;
//};
//
//template<typename ...U, typename TFun> Field<U...> &
//assign(Field<U...> &f, mesh_as::EntityRange const &r0, TFun const &op,
//      CHECK_FUNCTION_SIGNATURE(typename Field<U...>::value_type, TFun(typename Field<U...>::value_type & )))
//{
//    f.deploy();
//
//    auto const &m = *f.mesh_as();
//
//    static const mesh_as::MeshEntityType IFORM = Field<U...>::iform;
//
//    r0.foreach([&](mesh_as::MeshEntityId const &s) { op(f[s]); });
//
//    return f;
//}
//
//template<typename ...U, typename ...V> Field<U...> &
//assign(Field<U...> &f,mesh_as::EntityRange const &r0,  Field<V...> const &g)
//{
//    f.deploy();
//
//    auto const &m = *f.mesh_as();
//
//    static const mesh_as::MeshEntityType IFORM = Field<U...>::iform;
//
//    r0.foreach([&](mesh_as::MeshEntityId const &s) { f[s] = g[s]; });
//
//    return f;
//}


//template<typename V, typename M, MeshEntityType IFORM> constexpr MeshEntityType Patch<V, M, IFORM>::iform = IFORM;
//
//**
// *  PlaceHolder class of PatchBase
// */
//struct PatchBase : public toolbox::Object, public Acceptor
//{
//    SP_OBJECT_HEAD(PatchBase, toolbox::Object)
//
//public:
//
//    PatchBase() { }
//
//    virtual PatchBase }
//
//    PatchBase(PatchBase const &other) = delete;
//
//    PatchBase(PatchBase &&other) = delete;
//
//    PatchBase &operator=(PatchBase const &) = delete;
//
//    void swap(PatchBase &other) = delete;
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
//    std::shared_ptr<TF> add(MeshBlock const *m, Args &&...args)
//    {
//        assert(m != nullptr);
//
//        std::shared_ptr<TF> res;
//
//        static_assert(std::is_base_of<View, TF>::entity,
//                      "Object is not a get_mesh::PatchBase::View");
//        auto it = m_views_.find(m->uuid());
//
//        if (it != m_views_.end())
//        {
//
//            if (!it->second->template is_a<TF>())
//            {
//                RUNTIME_ERROR << "PatchBase type cast error! "
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
//            RUNTIME_ERROR << "MeshBlock [" << boost::uuids::hash_value(id) << "] is missing!" << std::endl;
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
