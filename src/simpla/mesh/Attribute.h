//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Printable.h>

#include "MeshBlock.h"
#include "DataBlock.h"
#include "Worker.h"

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
        public std::enable_shared_from_this<Attribute>
{


public:

    SP_OBJECT_HEAD(Attribute, Object)

    Attribute(std::string const &s = "");

    Attribute(Attribute const &) = delete;

    Attribute(Attribute &&) = delete;

    virtual ~Attribute();

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual std::string name() const { return Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(const data::DataBase &);

    virtual void save(data::DataBase *) const;

    virtual bool has(MeshBlock const *) const;

    virtual void insert(MeshBlock const *m, const std::shared_ptr<DataBlock> &);

    virtual void erase(MeshBlock const *);

    virtual DataBlock const *at(MeshBlock const *m) const;

    virtual DataBlock *at(const MeshBlock *);


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template<typename TV, MeshEntityType IFORM>
class AttributeProxy : public Attribute
{
public:
    template<typename ...Args>
    AttributeProxy(Args &&...args):Attribute(std::forward<Args>(args)...) {}

    virtual ~AttributeProxy() {}

    virtual MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(TV); };

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
template<typename TV, MeshEntityType IFORM>
class AttributeView : public Worker::Observer
{
protected:
    typedef AttributeView this_type;
    typedef AttributeProxy<TV, IFORM> attribute_type;
    std::shared_ptr<attribute_type> m_attr_;
    MeshBlock const *m_mesh_;
    DataBlock *m_data_;
public:


    AttributeView(Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new attribute_type()) {};

    AttributeView(std::string const &s, Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new attribute_type(s)) {};

    AttributeView(MeshBlock const *m, std::string const &s = "", Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new attribute_type(s)), m_mesh_(m) {};

    AttributeView(std::shared_ptr<Attribute> const &attr, Worker *w) :
            Worker::Observer(w), m_attr_(attr) {};


    virtual ~AttributeView() {}

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    attribute_type *attribute() { return m_attr_.get(); }

    attribute_type const *attribute() const { return m_attr_.get(); }

    MeshBlock const *mesh() const { return m_mesh_; };

    DataBlock *data() { return m_data_; }

    DataBlock const *data() const { return m_data_; }

    virtual MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(TV); };

    virtual std::string name() const { return m_attr_->name(); }

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        if (m_mesh_ != nullptr && m_data_ != nullptr) { m_data_->print(os, indent + 1); }
        else { os << "not-deployed!"; }
        return os;
    }

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }

    virtual std::shared_ptr<DataBlock> clone(MeshBlock const *m) const =0;

    /**
     * move to block m;
     *   if m_attr_.at(m) ==nullptr then  m_attr_.insert(m_data_.clone(m))
     *   m_data_= m_attr_.at(m)
     *
     * @param m
     * @result
     *  m_mesh_ : m
     *  m_data_ : not nullptr. m_attr_.at(m) ;
     */
    virtual void move_to(MeshBlock const *m)
    {
        ASSERT (m != nullptr)
        if (m != m_mesh_)
        {
            m_mesh_ = m;

            try
            {
                m_data_ = m_attr_->at(m_mesh_);

            }
            catch (std::out_of_range const &error)
            {
                auto res = clone(m_mesh_);
                m_attr_->insert(m_mesh_, res);
                m_data_ = res.get();
            }
        }
    }

    /**
      *  erase data from attribute
      *
      *   m_attr_.erase(m)
      *
      * @note do not destroy m_data_
      *
      * @result
      *   m_data_ : nullptr
      *   m_mesh_ : nullptr
      */
    virtual void erase()
    {
        ASSERT (m_attr_ != nullptr);

        m_attr_->erase(m_mesh_);
        m_mesh_ = nullptr;
    }

    /**
     *  malloc data at current block
     *  @result
     *    m_mesh_ : not chanaged
     *    m_data_ : is_deployed()=true
     */
    virtual void deploy()
    {
        if (m_data_ == nullptr) { move_to(m_mesh_); }
        m_data_->deploy();
    }

    /**
     * release data memory at current block
     * @result
     *   m_mesh_ : not change
     *   m_data_ : is_deployed()=false
     */
    virtual void destroy() { if (m_data_ != nullptr) { m_data_->destroy(); }};

    /**
     *  if m_attr_.has(other) then m_data_.copy(m_attr_.at(other),only_ghost)
     *  else do nothing
     * @param other
     * @param only_ghost
     */
    virtual void sync(MeshBlock const *other, bool only_ghost = true)
    {
        try { m_data_->sync(m_attr_->at(other), only_ghost); } catch (std::out_of_range const &) {}
    };


};


}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

