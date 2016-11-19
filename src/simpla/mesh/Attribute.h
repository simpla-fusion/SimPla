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

//@formatter:off
typedef design_pattern::Observer<void(std::shared_ptr<MeshBlock>const & )> mesh_observer_type;
//@formatter:on
struct AttributeViewBase : public mesh_observer_type
{
    typedef design_pattern::Observable<void(std::shared_ptr<MeshBlock> const &)> observable;

    AttributeViewBase(observable *w = nullptr) { if (w != nullptr) { w->connect(this); }};

    virtual ~AttributeViewBase() {}

    AttributeViewBase(AttributeViewBase const &other) = delete;

    AttributeViewBase(AttributeViewBase &&other) = delete;

    virtual Attribute *attribute() =0;

    virtual Attribute const *attribute() const =0;

    virtual MeshBlock const *mesh_block() const =0;

    virtual DataBlock *data()=0;

    virtual DataBlock const *data() const =0;

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual bool is_a(std::type_info const &t_info) const =0;

    virtual void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)=0;

    virtual void notify(std::shared_ptr<MeshBlock> const &m) { move_to(m, nullptr); };

    virtual void deploy() {};

    virtual void destroy() {};
};

/**
 * AttributeView: expose one block of attribute
 * * is a view of Attribute
 * * is unaware of the type of Mesh
 * * has a pointer to a mesh block
 * * has a pointer to a data block
 * * has a shared pointer of attribute
 * * can traverse on the Attribute
 * *
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

    AttributeView() {}
//    template<typename ...Args>
//    explicit AttributeView(Args &&...args):m_attr_(new attribute_type(std::forward<Args>(args)...)) {};

    template<typename ...Args>
    explicit AttributeView(observable *w, Args &&...args)
            :AttributeViewBase(w), m_attr_(new attribute_type(std::forward<Args>(args)...)) {};

    AttributeView(std::shared_ptr<Attribute> const &attr) : m_attr_(attr) {};

    virtual ~AttributeView() {}

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    attribute_type *attribute() { return m_attr_.get(); }

    attribute_type const *attribute() const { return m_attr_.get(); }

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


};


}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

