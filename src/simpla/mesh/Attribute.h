//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Serializable.h>
#include <simpla/toolbox/Printable.h>

#include "MeshBlock.h"
#include "DataBlock.h"
#include "Worker.h"

namespace simpla { namespace mesh
{

/**
 *  AttributeBase IS-A container of data blocks
 */
class Attribute :
        public toolbox::Object,
        public toolbox::Printable,
        public toolbox::Serializable,
        public std::enable_shared_from_this<Attribute>
{


public:

    SP_OBJECT_HEAD(Attribute, toolbox::Object)

    Attribute(std::string const &s = "");

    Attribute(Attribute const &) = delete;

    Attribute(Attribute &&) = delete;

    virtual ~Attribute();

    virtual std::string name() const { return toolbox::Object::name(); };

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

class AttributeView : public Worker::Observer
{


protected:
    typedef AttributeView this_type;
    std::shared_ptr<Attribute> m_attr_;
    MeshBlock const *m_mesh_;
    DataBlock *m_data_;

public:


    AttributeView(Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new Attribute()) {};

    AttributeView(std::string const &s, Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new Attribute(s)) {};

    AttributeView(MeshBlock const *m, std::string const &s = "", Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new Attribute(s)), m_mesh_(m) {};

    AttributeView(std::shared_ptr<Attribute> const &attr, Worker *w) :
            Worker::Observer(w), m_attr_(attr) {};


    virtual ~AttributeView() {}

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    std::shared_ptr<Attribute> &attribute() { return m_attr_; }

    MeshBlock const *mesh() const { return m_mesh_; };

    DataBlock *data() { return m_data_; }

    DataBlock const *data() const { return m_data_; }

    virtual std::string name() const;

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

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
    virtual void move_to(MeshBlock const *m);

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
    virtual void erase();

    /**
     *  malloc data at current block
     *  @result
     *    m_mesh_ : not chanaged
     *    m_data_ : is_deployed()=true
     */
    virtual void deploy();

    /**
     * release data memory at current block
     * @result
     *   m_mesh_ : not change
     *   m_data_ : is_deployed()=false
     */
    virtual void destroy();

    /**
     *  if m_attr_.has(other) then m_data_.copy(m_attr_.at(other),only_ghost)
     *  else do nothing
     * @param other
     * @param only_ghost
     */
    virtual void sync(MeshBlock const *other, bool only_ghost = true);


};

}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

