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
#include "MeshEntity.h"
#include "MeshAtlas.h"


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

    void apply(Visitor &vistor) { vistor.visit(*this); };

    void apply(Visitor &vistor) const { vistor.visit(*this); };


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

        virtual void swap(View &other) { }

        virtual MeshEntityType entity_type() const = 0;

        virtual MeshEntityRange const &range() const = 0;

        virtual data_model::DataSet get_dataset() const = 0;

        virtual void set_dataset(data_model::DataSet const &) = 0;
    };

    /** register MeshBlockId to attribute data collection.  */




    template<typename TF, typename ...Args>
    std::shared_ptr<TF> add(MeshBase const *m, Args &&...args)
    {
        assert(m != nullptr);

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

            return std::make_shared<TF>(*std::dynamic_pointer_cast<TF>(it->second));
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
                auto ptr = std::make_shared<TF>(m, std::forward<Args>(args)...);

                m_attrs_.emplace(std::make_pair(m->uuid(), std::dynamic_pointer_cast<View>(ptr)));

                return ptr;
            }
        }


    }


    /** erase MeshBlockId from attribute data collection.  */
    size_t erase(MeshBlockId const &id)
    {
        return m_attrs_.erase(id);
    }

    data_model::DataSet get_dataset(MeshBlockId const &id) const
    {
        try
        {
            return m_attrs_.at(id)->get_dataset();

        }
        catch (std::out_of_range const &)
        {
            RUNTIME_ERROR << "Block [" << boost::uuids::hash_value(id) << "] is missing!" << std::endl;
        }

    }

    void set_dataset(MeshBlockId const &id, data_model::DataSet const &d)
    {
        try
        {
            return m_attrs_.at(id)->set_dataset(d);

        }
        catch (std::out_of_range const &)
        {
            RUNTIME_ERROR << "Block [" << boost::uuids::hash_value(id) << "] is missing!" << std::endl;
        }
    }

    void get_dataset(std::map<MeshBlockId, data_model::DataSet> *res) const
    {
        for (auto const &item:m_attrs_)
        {
            res->emplace(std::make_pair(item.first, item.second->get_dataset()));
        };
    }

    void set_dataset(std::map<MeshBlockId, data_model::DataSet> const &d)
    {
        for (auto const &item:d) { set_dataset(item.first, item.second); }
    }

protected:
    std::map<MeshBlockId, std::shared_ptr<View>> m_attrs_;
};

//
///**
// *  Data attached to mesh entities.
// *
// *  @comment similar to MOAB::Tag
// *
// **/
//template<typename ...> class MeshAttribute;
//
//template<typename TV, typename TM, size_t IEntityType>
//class MeshAttribute<TV, TM, index_const<IEntityType>, tags::DENSE>
//        : public MeshAttribute
//{
//private:
//    typedef MeshAttribute<TV, TM, index_const<IEntityType>, tags::DENSE> this_type;
//
//    typedef MeshAttribute<TV, TM, index_const<IEntityType>, tags::DENSE> mesh_attribute_type;
//
//
//public:
//    typedef TV value_type;
//    typedef TM mesh_type;
//
//    //*************************************************
//    // Object Head
//
//private:
//    typedef MeshAttribute base_type;
//public:
//    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info || base_type::is_a(info); }
//
//    virtual std::string get_class_name() const { return class_name(); }
//
//    static std::string class_name()
//    {
//        return std::string("MeshAttribute<") +
//               traits::type_id<TV, TM, index_const<IEntityType>>::name() + ">";
//    }
//    //*************************************************
//public:
//
//    class View;
//
//    MeshAttribute(MeshAtlas const &m) : MeshAttribute(m) { };
//
//    virtual  ~MeshAttribute() { };
//
//    MeshAttribute(MeshAttribute const &other) = delete;
//
//    MeshAttribute(MeshAttribute &&other) = delete;
//
//    MeshAttribute &operator=(MeshAttribute const &) = delete;
//
//    void swap(MeshAttribute &other) = delete;
//
//    virtual MeshEntityType entity_type() const { return static_cast<MeshEntityType >(IEntityType); }
//
//
//    /** register MeshBlockId to attribute data collection.  */
//    virtual bool add(MeshBlockId const &m_id)
//    {
//        bool success = false;
//        auto it = m_data_collection_.find(m_id);
//        if (it == m_data_collection_.end())
//        {
//            if (!m_atlas_.has(m_id))
//            {
//                RUNTIME_ERROR << "Mesh is not regitered! [" << hash_value(m_id) << "]" << std::endl;
//            }
//
//            size_t m_size = m_atlas_.template at<mesh_type>(m_id)->size(entity_type());
//
//            // @NOTE  !!! HERE allocate memory !!!
//            std::tie(std::ignore, success) =
//                    m_data_collection_.emplace(
//                            std::make_pair(m_id,
//                                           sp_alloc_array<value_type>(m_size)));
//        }
//        return success;
//    };
//
//    /** remove MeshBlockId from attribute data collection.  */
//    virtual void remove(MeshBlockId const &m_id) { m_data_collection_.erase(m_id); };
//
//private:
//
//    typedef std::map<MeshBlockId, std::shared_ptr<value_type> > data_collection_type;
//    data_collection_type m_data_collection_;
//
//public:
//
//    struct View : public MeshAttribute::View
//    {
//
//        typedef typename MeshAttribute::View base_type;
//        typedef mesh_attribute_type host_type;
//
//        View() { }
//
//        View(mesh_type const *m = nullptr, value_type *d = nullptr)
//                : base_type(m != nullptr ? m->uuid() : MeshBlockId()), m_mesh_(m), m_data_(d),
//                  m_range_(m != nullptr ? m->range(static_cast<MeshEntityType>(IEntityType)) : MeshEntityRange()) { }
//
//        View(View const &other) : base_type(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_),
//                                  m_range_(other.m_range_) { }
//
//        View(View &&other) : base_type(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_),
//                             m_range_(other.m_range_) { }
//
//        virtual  ~View() { }
//
//        virtual void swap(MeshAttribute::View &other)
//        {
//            base_type::swap(other);
//            auto &o_view = static_cast<View &>(other);
//            std::swap(m_mesh_, o_view.m_mesh_);
//            std::swap(m_data_, o_view.m_data_);
//            m_range_.swap(o_view.m_range_);
//        }
//
//        virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(View); }
//
//        virtual MeshEntityType entity_type() const { return static_cast<MeshEntityType >(IEntityType); }
//
//        virtual MeshEntityRange const &range() const { return m_range_; }
//
//        virtual MeshEntityRange &range() { return m_range_; }
//
//        mesh_type const &mesh() const { return *m_mesh_; }
//
//        inline value_type &get(MeshEntityId const &s) { return m_data_[m_mesh_->hash(s)]; }
//
//        inline value_type const &get(MeshEntityId const &s) const { return m_data_[m_mesh_->hash(s)]; }
//
//        inline value_type &operator[](MeshEntityId const &s) { return get(s); }
//
//        inline value_type const &operator[](MeshEntityId const &s) const { return get(s); }
//
//    private:
//        mesh_type const *m_mesh_;
//        value_type *m_data_;
//        MeshEntityRange m_range_;
//
//    };
//
//
//    View view(MeshBlockId const &id)
//    {
//        MeshBlockId m_id = (id.is_nil()) ? m_atlas_.root() : id;
//
//        add(m_id);
//        return View(m_id, m_atlas_.template at<mesh_type>(m_id), m_data_collection_[m_id].get());
//
//    }
//
//    virtual std::shared_ptr<MeshAttribute::View> view_(MeshBlockId const &id)
//    {
//        MeshBlockId m_id = (id.is_nil()) ? m_atlas_.root() : id;
//
//        add(m_id);
//
//        return std::dynamic_pointer_cast<MeshAttribute::View>(
//                std::make_shared<View>(m_id, m_atlas_.template at<mesh_type>(m_id),
//                                       m_data_collection_[m_id].get()));
//    }
//
//}; // class MeshAttribute

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESHATTRIBUTE_H
