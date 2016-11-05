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
 *  AttributeBase IS-A container of datablock
 */
class Attribute :
        public toolbox::Object,
        public toolbox::Printable,
        public toolbox::Serializable,
        public std::enable_shared_from_this<Attribute>
{


public:

    SP_OBJECT_HEAD(Attribute, toolbox::Object)

    Attribute(std::string const &s);

    Attribute(Attribute const &) = delete;

    Attribute(Attribute &&) = delete;

    virtual ~Attribute();

    virtual std::string name() const { return toolbox::Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(const data::DataBase &);

    virtual void save(data::DataBase *) const;

    void insert(MeshBlock const *m, const std::shared_ptr<DataBlock> &);

    virtual bool has(MeshBlock const *) const;

    virtual void erase(MeshBlock const *);

    virtual void deploy(MeshBlock const * = nullptr);

    virtual void clear(MeshBlock const * = nullptr);

    virtual void update(MeshBlock const *, MeshBlock const * = nullptr);

    virtual DataBlock const *at(MeshBlock const *m = nullptr) const;

    virtual DataBlock *at(const MeshBlock *, const MeshBlock *hint = nullptr);

    template<typename TB>
    TB *as(MeshBlock const *m)
    {
        if (!has(m))
        {
            auto res = std::make_shared<TB>(m);
            insert(m, std::dynamic_pointer_cast<DataBlock>(res));
            return res.get();

        } else
        {
            return static_cast<TB *>(at(m));
        }
    };

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};


class AttributeView :
        public Worker::Observer,
        public toolbox::Printable
{
    std::shared_ptr<mesh::Attribute> m_attr_;
    MeshBlock const *m_mesh_ = nullptr;
public:

    AttributeView(MeshBlock *m = nullptr, std::string const &s = "", Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new mesh::Attribute(s)), m_mesh_(m) {};

    template<typename TM>
    AttributeView(std::shared_ptr<TM> const &m, std::string const &s = "", Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new mesh::Attribute(s)), m_mesh_(m.get()) {};


    AttributeView(std::string const &s, Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(new mesh::Attribute(s)) {};

    AttributeView(std::shared_ptr<mesh::Attribute> const &attr, Worker *w = nullptr) :
            Worker::Observer(w), m_attr_(attr) {};

    virtual ~AttributeView() {}

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;


    virtual mesh::MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;


    template<typename TM> TM const *mesh() const { return static_cast<TM const *>(m_mesh_); }

    template<typename TB> TB *data() const { return m_attr_->as<TB>(m_mesh_); }

    MeshBlock const *mesh() const { return m_mesh_; }

    DataBlock *data() const { return m_attr_->at(m_mesh_); }

    std::shared_ptr<mesh::Attribute> &attribute() { return m_attr_; }

    virtual std::string name() const { return m_attr_->name(); };

    virtual std::ostream &print(std::ostream &os, int indent) const { return m_attr_->print(os, indent); }


    virtual void create(MeshBlock const *m, bool is_scratch = false) { UNIMPLEMENTED; };

    virtual void destroy() { UNIMPLEMENTED; };

    virtual void deploy(MeshBlock const *m = nullptr) { if (m != nullptr) { m_mesh_ = m; }};

    virtual void move_to(MeshBlock const *m = nullptr) { if (m != nullptr) { m_mesh_ = m; }};

    virtual void erase(MeshBlock const *m = nullptr) { UNIMPLEMENTED; };

    virtual void update(MeshBlock const *m = nullptr, bool only_ghost = false) { UNIMPLEMENTED; };
};

}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

