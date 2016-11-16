//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include <vector>
#include <set>
#include <simpla/data/DataBase.h>
#include <simpla/toolbox/Log.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include "MeshCommon.h"

namespace simpla { namespace mesh
{
struct MeshBlock;
struct DataBlock;
struct Attribute;

class Worker :
        public Object,
        public concept::Printable,
        public concept::Serializable
{
public:
    SP_OBJECT_HEAD(Worker, Object)
    struct Observer;
    struct Visitor;

    Worker();

    ~Worker();


    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual std::string name() const { return m_name_; };

    virtual void load(data::DataBase const &) { UNIMPLEMENTED; }

    virtual void save(data::DataBase *) const { UNIMPLEMENTED; }

    void attach(Observer *);

    void detach(Observer *);

    /**
      *  move '''Observer''' to mesh block '''m'''
      *  if  data block is not exist, create
      * @param id
      */
    void move_to(const std::shared_ptr<MeshBlock> &m);

    MeshBlock const *mesh() const;

    virtual std::shared_ptr<mesh::MeshBlock>
    create_mesh_block(index_type const *lo, index_type const *hi, Real const *dx,
                      Real const *xlo = nullptr, Real const *xhi = nullptr) const =0;

    virtual void initialize(Real data_time)=0;

    virtual void next_time_step(Real data_time, Real dt)=0;


    virtual void setPhysicalBoundaryConditions(double time)=0;

    /**
      *  deploy data on the mesh block   '''m'''
      *  if m==nullptr then deploy date on the current block
      * @param id
      */
    void deploy();

    /**
     *  destroy data on current mesh block,
     */
    void destroy();

    /**
     * copy/refine/coarsen data from '''other''' block to current block
     *  copy   : current_block.level == other_block.level
     *  refine : current_block.level >  other_block.level
     *  coarsen: current_block.level <  other_block.level
     *
     *  require '''current data block''' is not created
     */
    void sync(MeshBlock const *other, bool only_ghost = true);

    void apply(Visitor const &);

    void apply(Visitor const &) const;

    void for_each(std::function<void(Observer &)> const &);

    void for_each(std::function<void(Observer const &)> const &) const;


private:
    std::string m_name_;
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

struct Worker::Visitor
{
    virtual void visit(Observer &)=0;

    virtual void visit(Observer const &) const =0;
};

struct Worker::Observer : public concept::Printable
{

    Observer(Worker *m);

    virtual ~Observer();

    Observer(Observer const &other) = delete;

    Observer(Observer &&other) = delete;

    virtual std::string name() const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;

    virtual Attribute *attribute() { return nullptr; };

    virtual Attribute const *attribute() const { return nullptr; };
//    virtual void move_to(MeshBlock const *) =0;
//
//    virtual void erase() =0;
//
//    virtual void deploy() =0;
//
//    virtual void destroy() =0;
//
//    virtual void sync(MeshBlock const *other, bool only_ghost = true) =0;


    /**
     * move to block m;
     *   if m_attr_.at(m) ==nullptr then  m_attr_.insert(m_data_.clone(m))
     *   m_data_= m_attr_.at(m)
     *
     * @param m
     * @result
     *  m_mesh_ : m
     *  m_data_ : m_attr_.at(m) ;
     */
    virtual void move_to(const std::shared_ptr<MeshBlock> &m)=0;

    virtual void move_to(const std::shared_ptr<MeshBlock> &m, const std::shared_ptr<DataBlock> &d)=0;


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
    virtual void erase(MeshBlock const *m = nullptr)=0;

    /**
     *  malloc data at current block
     *  @result
     *    m_mesh_ : not chanaged
     *    m_data_ : is_deployed()=true
     */
    virtual void deploy()=0;

    /**
     * release data memory at current block
     * @result
     *   m_mesh_ : not change
     *   m_data_ : is_deployed()=false
     */
    virtual void destroy()=0;

    /**
     *  if m_attr_.has(other) then m_data_.copy(m_attr_.at(other),only_ghost)
     *  else do nothing
     * @param other
     * @param only_ghost
     */
    virtual void sync(MeshBlock const *other, bool only_ghost = true)=0;


private:
    friend class Worker;

    Worker *m_worker_;
};
}}
#endif //SIMPLA_WORKER_H
