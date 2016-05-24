/**
 * @file MeshAttribute.h
 * @author salmon
 * @date 2016-05-19.
 */

#ifndef SIMPLA_MESHATTRIBUTE_H
#define SIMPLA_MESHATTRIBUTE_H

#include "../base/Object.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshAtlas.h"
#include "../gtl/Log.h"

namespace simpla { namespace io { class IOHandle; }}
namespace simpla { namespace mesh
{
namespace tags
{
struct DENSE;
struct SPARSE;
struct BIT;
}
class MeshAtlas;


/**
 *  Base class of MeshAttributeBase
 */
struct MeshAttributeBase : public base::Object
{
    SP_OBJECT_HEAD(MeshAttributeBase, base::Object)

    MeshAttributeBase(MeshAtlas const &m) : m_atlas_(m) { }

    virtual ~MeshAttributeBase() { }

    MeshAttributeBase(MeshAttributeBase const &other) = delete;

    MeshAttributeBase(MeshAttributeBase &&other) = delete;

    MeshAttributeBase &operator=(MeshAttributeBase const &) = delete;

    void swap(MeshAttributeBase &other) = delete;


//    virtual bool is_a(std::type_info const &t_info) const = 0;

    template<typename T> inline bool is_a() const
    {
        return (std::is_base_of<MeshAttributeBase, T>::value && is_a(typeid(T)));
    };

    virtual MeshEntityType entity_type() const = 0;

    /** read attribute data on m_id from IOHandle, read all if m_id=0*/
    virtual void read(io::IOHandle &, MeshBlockId m_id = 0) = 0;

    /** write attribute data on m_id to IOHandle, write all if m_id=0*/
    virtual void write(io::IOHandle &, MeshBlockId m_id = 0) const = 0;


    /** register MeshBlockId to attribute data collection.  */
    virtual bool add(MeshBlockId) = 0;

    /** remove MeshBlockId from attribute data collection.  */
    virtual void remove(MeshBlockId) = 0;


    struct View
    {
        View(MeshBlockId m_id = 0) : m_id_(m_id) { }

        View(View const &other) : m_id_(other.m_id_) { }

        View(View &&other) : m_id_(other.m_id_) { }

        ~View() { };

        View &operator=(View const &other) = delete;

        virtual bool is_a(std::type_info const &t_info) const = 0;

        template<typename T> inline bool is_a() const { return (std::is_base_of<View, T>::value && is_a(typeid(T))); }

        virtual void swap(View &other) { std::swap(m_id_, other.m_id_); }

        virtual MeshEntityType entity_type() const = 0;

        virtual EntityRange range() const = 0;

        MeshBlockId block_id() const { return m_id_; }

    private:
        MeshBlockId m_id_;

    };

    virtual std::shared_ptr<View> view_(MeshBlockId id) = 0;


protected:
    MeshAtlas const &m_atlas_;
};

/**
 *  Data attached to mesh entities.
 *
 *  @comment similar to MOAB::Tag
 *
 **/
template<typename ...> class MeshAttribute;

template<typename TV, typename TM, int IEntityType>
class MeshAttribute<TV, TM, std::integral_constant<int, IEntityType>, tags::DENSE>
        : public MeshAttributeBase,
          public std::enable_shared_from_this<MeshAttribute<TV, TM, std::integral_constant<int, IEntityType>, tags::DENSE> >
{
    typedef MeshAttribute<TV, TM, std::integral_constant<int, IEntityType>, tags::DENSE> this_type;

    typedef MeshAttribute<TV, TM, std::integral_constant<int, IEntityType>, tags::DENSE> mesh_attribute_type;

public:
    typedef TV value_type;
    typedef TM mesh_type;
private:
    typedef std::map<MeshBlockId, std::shared_ptr<value_type> > data_collection_type;
public:
    class View;

    MeshAttribute(MeshAtlas const &m) : MeshAttributeBase(m) { };

    MeshAttribute(MeshAttribute const &other) = delete;

    MeshAttribute(MeshAttribute &&other) = delete;

    MeshAttribute &operator=(MeshAttribute const &) = delete;

    void swap(MeshAttribute &other) = delete;


    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }


    virtual MeshEntityType entity_type() const { return static_cast<MeshEntityType >(IEntityType); }


    virtual void read(io::IOHandle &, MeshBlockId m_id) { UNIMPLEMENTED; };

    virtual void write(io::IOHandle &, MeshBlockId m_id) const { UNIMPLEMENTED; };


    /** register MeshBlockId to attribute data collection.  */
    virtual bool add(MeshBlockId m_id)
    {
        if (m_data_collection_ == nullptr)
        {
            m_data_collection_ = std::make_shared<data_collection_type>();
        }

        bool success = false;

        auto it = m_data_collection_->find(m_id);
        if (it == m_data_collection_->end())
        {
            std::tie(it, success) =
                    m_data_collection_->emplace(
                            std::make_pair(m_id, std::make_shared<value_type>(
                                    m_atlas_.template at<mesh_type>(m_id)->size(entity_type()))));

        }
        return success;
    };

    /** remove MeshBlockId from attribute data collection.  */
    virtual void remove(MeshBlockId m_id) { m_data_collection_->erase(m_id); };


//    template<typename TFun, typename ...Args>
//    void accept(TFun const &fun, Args &&...args)
//    {
//        for (auto const &s:range()) { fun(get(s), std::forward<Args>(args)...); }
//    }

private:
    std::shared_ptr<data_collection_type> m_data_collection_;

public:
    View view(MeshBlockId m_id = 0)
    {
        if (m_id == 0) { m_id = m_atlas_.root(); }
        add(m_id);

        return std::make_shared<View>(m_id, m_atlas_.template at<mesh_type>(m_id).get(),
                                      (*m_data_collection_)[m_id].get());
    }


    struct View : public MeshAttributeBase::View
    {

        typedef typename MeshAttributeBase::View base_type;
        typedef mesh_attribute_type host_type;

        View(MeshBlockId id = 0, mesh_type const *m = nullptr, value_type *d = nullptr)
                : base_type(id), m_mesh_(m), m_data_(d) { };

        View(View const &other) : base_type(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_) { }

        View(View &&other) : base_type(other), m_mesh_(other.m_mesh_), m_data_(other.m_data_) { }

        virtual  ~View() { }

        virtual void swap(View &other)
        {
            base_type::swap(other);
            std::swap(m_mesh_, other.m_mesh_);
            std::swap(m_data_, other.m_data_);
        }

        virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(View); }

        MeshEntityType entity_type() const { return static_cast<MeshEntityType >(IEntityType); }

        EntityRange range() const { return m_mesh_->range(entity_type()); }

        mesh_type const &mesh() const { return *m_mesh_; }

        inline value_type &get(MeshEntityId const &s) { return m_data_[m_mesh_->hash(s)]; }

        inline value_type const &get(MeshEntityId const &s) const { return m_data_[m_mesh_->hash(s)]; }

        inline value_type &operator[](MeshEntityId const &s) { return get(s); }

        inline value_type const &operator[](MeshEntityId const &s) const { return get(s); }

    private:
        mesh_type const *m_mesh_;
        value_type *m_data_;

    };
}; // class MeshAttribute

}}//namespace simpla{namespace mesh{

#endif //SIMPLA_MESHATTRIBUTE_H
