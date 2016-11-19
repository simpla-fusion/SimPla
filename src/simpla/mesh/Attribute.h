//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Configurable.h>
#include <simpla/toolbox/design_pattern/Observer.h>

#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla { namespace mesh
{

/**
 *  AttributeBase IS-A container of data blocks
 *  Define of attribute
 *  *  is printable
 *  *  is serializable
 *  *  is unawar of mesh block
 *  *  is unawar of mesh atlas
 *  *  has a value type
 *  *  has a MeshEntityType (entity_type)
 *  *  has n data block (insert, erase,has,at)
 *  *
 */
class Attribute :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable,
        public std::enable_shared_from_this<Attribute>
{


public:

    SP_OBJECT_HEAD(Attribute, Object)

    Attribute(std::string const &s, std::string const &config_str = "");

    Attribute(Attribute const &) = delete;

    Attribute(Attribute &&) = delete;

    virtual ~Attribute();

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    // degree of freedom
    virtual size_t dof() const { return 1; };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::string name() const { return m_name_; };

    virtual void load(const data::DataBase &);

    virtual void save(data::DataBase *) const;

    virtual bool has(std::shared_ptr<MeshBlock> const &) const;

    virtual void erase(std::shared_ptr<MeshBlock> const &);

    virtual std::shared_ptr<DataBlock> const &at(std::shared_ptr<MeshBlock> const &m) const;

    virtual std::shared_ptr<DataBlock> &at(std::shared_ptr<MeshBlock> const &m);

    virtual std::shared_ptr<DataBlock> &
    get(std::shared_ptr<MeshBlock> const &, std::shared_ptr<DataBlock> const &p = nullptr);

    virtual void insert(const std::shared_ptr<MeshBlock> &m, const std::shared_ptr<DataBlock> &p = nullptr);

    virtual void register_data_block_factory(std::type_index idx,
                                             const std::function<std::shared_ptr<DataBlock>(
                                                     std::shared_ptr<MeshBlock> const &, void *)> &f);

private:
    std::string m_name_;
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template<typename TV, MeshEntityType IFORM, size_type IDOF>
class AttributeProxy : public Attribute
{
public:
    static constexpr mesh::MeshEntityType iform = IFORM;
    static constexpr size_type DOF = IDOF;

    template<typename ...Args>
    AttributeProxy(Args &&...args):Attribute(std::forward<Args>(args)...) {}

    virtual ~AttributeProxy() {}

    virtual MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(typename traits::value_type<TV>::type); };

    virtual size_type dof() const { return DOF; };

};

struct AttributeViewBase;

struct AttributeHolder;

struct AttributeViewBase : public design_pattern::Observer<void(std::shared_ptr<MeshBlock> const &)>
{
    typedef design_pattern::Observer<void(std::shared_ptr<MeshBlock> const &)> base_type;

    AttributeViewBase() {};

    virtual ~AttributeViewBase() {}

    AttributeViewBase(AttributeViewBase const &other) = delete;

    AttributeViewBase(AttributeViewBase &&other) = delete;

    virtual void notify(std::shared_ptr<MeshBlock> const &m) { move_to(m, nullptr); };

    virtual std::shared_ptr<Attribute> create_attribute(std::string const &key) const =0;

    virtual std::shared_ptr<Attribute> attribute() =0;

    virtual std::shared_ptr<Attribute> attribute() const =0;

    virtual void attribute(std::shared_ptr<Attribute> const &attr) =0;

    virtual MeshBlock const *mesh_block() const =0;

    virtual DataBlock *data()=0;

    virtual DataBlock const *data() const =0;

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual bool is_a(std::type_info const &t_info) const =0;

    virtual void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)=0;


    virtual void deploy() { DO_NOTHING; };

    virtual void destroy() { DO_NOTHING; };

    virtual void register_data_block_factory(std::shared_ptr<mesh::Attribute> const &attr)=0;

};


/**
 * AttributeView: expose one block of attribute
 * -) is a view of Attribute
 * -) is unaware of the type of Mesh
 * -) has a pointer to a mesh block
 * -) has a pointer to a data block
 * -) has a shared pointer of attribute
 * -) can traverse on the Attribute
 * -) if there is no Atlas, AttributeView will hold the MeshBlock
 * -) if there is no AttributeHolder, AttributeView will hold the DataBlock and Attribute
 */
template<typename TV, MeshEntityType IFORM, size_type IDOF = 1>
class AttributeView : public AttributeViewBase
{

protected:

    typedef AttributeView this_type;
    typedef AttributeProxy<TV, IFORM, IDOF> attribute_type;
    std::shared_ptr<attribute_type> m_attr_;
    std::shared_ptr<MeshBlock> m_mesh_holder_;
    std::shared_ptr<DataBlock> m_data_holder_;

public:

    static constexpr MeshEntityType iform = IFORM;

    static constexpr size_type DOF = IDOF;

    AttributeView() : m_attr_(nullptr), m_mesh_holder_(nullptr), m_data_holder_(nullptr) {}

    template<typename ...Args>
    explicit AttributeView(AttributeHolder *w, std::string const &key, Args &&...args)
            : m_attr_(nullptr), m_mesh_holder_(nullptr), m_data_holder_(nullptr)
    {
        connect(w, key, std::forward<Args>(args)...);
    };

    template<typename ...Args>
    explicit AttributeView(std::string const &key, Args &&...args)
            : m_attr_(new attribute_type(key, std::forward<Args>(args)...)), m_mesh_holder_(nullptr),
              m_data_holder_(nullptr) {};

    virtual ~AttributeView() {}

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    template<typename ...Args> void connect(AttributeHolder *w, std::string const &key, Args &&...args);

    std::shared_ptr<Attribute>
    create_attribute(std::string const &key) const
    {
        return std::dynamic_pointer_cast<Attribute>(std::make_shared<attribute_type>(key));
    };

    std::shared_ptr<Attribute> attribute() { return m_attr_; }

    std::shared_ptr<Attribute> attribute() const { return m_attr_; }

    virtual void
    attribute(std::shared_ptr<Attribute> const &attr)
    {
        m_attr_ = std::dynamic_pointer_cast<attribute_type>(attr);
    };

    virtual void register_data_block_factory(std::shared_ptr<mesh::Attribute> const &attr)
    {
        attr->register_data_block_factory(
                std::type_index(typeid(MeshBlock)),
                [&](const std::shared_ptr<MeshBlock> &m, void *p)
                {
                    return m->template create_data_block<TV, IFORM>(p);
                });
    }


    MeshBlock const *mesh_block() const { return m_mesh_holder_.get(); };

    DataBlock *data() { return m_data_holder_.get(); }

    DataBlock const *data() const { return m_data_holder_.get(); }

    MeshEntityType entity_type() const { return IFORM; };

    std::type_info const &value_type_info() const { return typeid(TV); };

    size_type dof() const { return DOF; };

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }

    virtual void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
    {
        ASSERT(m != nullptr);
        m_mesh_holder_ = m;
        m_data_holder_ = m_attr_->get(m, d);
    }

    virtual void deploy()
    {
        ASSERT(m_mesh_holder_ != nullptr);
        if (m_data_holder_ == nullptr) { m_data_holder_ = m_attr_->get(m_mesh_holder_); }
        m_data_holder_->deploy();
        CHECK(m_attr_->name());
    };


};

struct AttributeHolder :
        public design_pattern::Observable<void(std::shared_ptr<MeshBlock> const &)>
{
    typedef design_pattern::Observable<void(std::shared_ptr<MeshBlock> const &)> base_type;

    AttributeHolder() {}

    virtual ~AttributeHolder() {}

    template<typename ...Args> bool
    connect(AttributeViewBase *view, std::string const &key)
    {
        ASSERT(view != nullptr);
        bool success = true;

        auto it = m_attr_holders_.find(key);

        if (it == m_attr_holders_.end() && view->attribute() == nullptr)
        {
            success = false;
        } else
        {
            base_type::connect(static_cast<observer_type *>(view));

            if (it != m_attr_holders_.end())
            {
                view->attribute(it->second);
            } else if (view->attribute() != nullptr)
            {
                m_attr_holders_.emplace(std::make_pair(key, view->attribute()));
            }
            success = true;

        }
        return success;
    }

    virtual void disconnect(AttributeViewBase *view)
    {
        if (view->attribute() != nullptr) { m_attr_holders_.erase(view->attribute()->name()); }

        base_type::disconnect(view);
    }

    void move_to(std::shared_ptr<MeshBlock> const &m) { notify(m); }

    virtual void remove(AttributeViewBase *observer) { base_type::remove(observer); }

    virtual void foreach(std::function<void(AttributeViewBase const &)> const &fun) const
    {
        base_type::foreach([&](observer_type const &obj) { fun(static_cast< AttributeViewBase const &>(obj)); });
    };

    virtual void foreach(std::function<void(AttributeViewBase &)> const &fun)
    {
        base_type::foreach([&](observer_type &obj) { fun(static_cast< AttributeViewBase &>(obj)); });
    };
private:
    std::map<std::string, std::shared_ptr<Attribute>> m_attr_holders_;
};


template<typename TV, MeshEntityType IFORM, size_type IDOF> template<typename ...Args> void
AttributeView<TV, IFORM, IDOF>::connect(AttributeHolder *w, std::string const &key, Args &&...args)
{
    if (!w->connect(this, key))
    {
        m_attr_ = std::make_shared<attribute_type>(key, std::forward<Args>(args)...);

        register_data_block_factory(m_attr_);

        if (!w->connect(this, key))
        {
            RUNTIME_ERROR << "Can not connect attribute view!" << std::endl;
        }
    }
};
}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

