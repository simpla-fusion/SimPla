//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <memory>

#include <simpla/data/DataBase.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/Object.h>
#include <simpla/toolbox/Printable.h>
#include <simpla/toolbox/Serializable.h>


namespace simpla { namespace mesh
{
struct MeshBlock;

class Worker :
        public toolbox::Object,
        public toolbox::Printable,
        public toolbox::Serializable
{
public:
    SP_OBJECT_HEAD(Worker, toolbox::Object);
    struct Observer;

    Worker();

    ~Worker();

    virtual std::string  name() const { return toolbox::Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual void load(data::DataBase const &) { UNIMPLEMENTED; }

    virtual void save(data::DataBase *) const { UNIMPLEMENTED; }

    void attach(Observer *);

    void detach(Observer *);

    /**
     *  destroy data on all attached Observers,
     */
    void destroy() const;

    /**
      *  move '''Observer''' to mesh block '''m'''
      *  if  data block is not exist, create
      * @param id
      */
    void move_to(mesh::MeshBlock const *m) const;

    /**
     * using  current data block create a new data block on the '''m'''
     * @param m
     */
    void create(mesh::MeshBlock const *m) const;

    /**
      *  deploy data on the mesh block   '''m'''
      *  if m==nullptr then deploy date on the current block
      * @param id
      */
    void deploy(mesh::MeshBlock const *m = nullptr) const;

    /**
     * erase '''Observer''' data on a mesh block
     * @param id id of mesh block
     */
    void erase(mesh::MeshBlock const *m = nullptr) const;


    /**
     * copy/refine/coarsen data from '''other''' block to current block
     *  copy   : current_block.level == other_block.level
     *  refine : current_block.level >  other_block.level
     *  coarsen: current_block.level <  other_block.level
     *
     *  require '''current data block''' is not created
     */
    void update(mesh::MeshBlock const *m = nullptr, bool only_ghost = false) const;

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


struct Worker::Observer
{

    Observer(Worker *m);

    virtual ~Observer();

    Observer(Observer const &other) = delete;

    Observer(Observer &&other) = delete;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;

    virtual void destroy() =0;

    virtual void create(mesh::MeshBlock const *, bool is_scratch = false) =0;

    virtual void deploy(mesh::MeshBlock const *) =0;

    virtual void move_to(mesh::MeshBlock const *) =0;

    virtual void erase(mesh::MeshBlock const *)=0;

    virtual void update(mesh::MeshBlock const *, bool only_ghost = false) =0;

private:
    friend class Worker;

    Worker *m_worker_;
};
}}
#endif //SIMPLA_WORKER_H
