//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DICT_H
#define SIMPLA_DICT_H


#include <simpla/SIMPLA_config.h>

#include <typeinfo>
#include <string>
#include <memory>
#include <ostream>
#include <iomanip>

namespace simpla { namespace toolbox
{

struct DataType;

struct DataSpace;


struct DataEntity
{
public:
    DataEntity() {}

    virtual ~DataEntity() {}

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataEntity); }

    virtual const std::type_info &type() const =0;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const =0;

    virtual bool is_null() const =0;

    virtual bool is_simple() const { return true; };

    virtual bool is_table() const { return false; };

    virtual bool is_array() const { return false; };


//        virtual DataType data_type(){return DataType();};
//
//        virtual DataSpace data_space() {return DataSpace();};

    virtual const void *data() const { return nullptr; };

    virtual void *data() { return nullptr; };
};


class DataBase : public DataEntity
{
public:

    DataBase() {};

    virtual  ~DataBase() {};

    //as data entity

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataBase) || DataEntity::is_a(t_id); }

    virtual bool is_table() const { return true; };

    virtual const std::type_info &type() const { return typeid(DataBase); };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual bool is_null() const { return size() == 0; };

    // as database

    virtual bool eval(std::string path) { return true; };

    virtual bool open(std::string path, int flag = 0) { return true; };

    virtual void close() {};

    virtual size_t size() const =0;

    virtual bool empty() const =0;

    virtual bool has(std::string const &key) const =0;


    /**
     *  as container
     */
    virtual void set(std::string const &, std::shared_ptr<DataEntity> const &) =0;

    /**
    *  if key exists then return ptr else create and return ptr
    * @param key
    * @return
    */
    virtual std::shared_ptr<DataEntity> get(std::string const &key)=0;

    /**
     *  if key exists then return ptr else return null
     * @param key
     * @return
     */
    virtual std::shared_ptr<DataEntity> at(std::string const &key)=0;

    virtual std::shared_ptr<const DataEntity> at(std::string const &key) const =0;

    /**
     *   foreach(<lambda>)  lets container decide the traversing algorithm,
     *   which is better than iterator to traverse containers.
     *   *. easy to overload, need  not to implement iterator class
     *   *. easy to parallism
     *
     */

    virtual void foreach(std::function<void(std::string const &, DataEntity &)> const &fun)=0;

    virtual void foreach(std::function<void(std::string const &, DataEntity const &)> const &fun) const =0;


};

std::ostream &operator<<(std::ostream &os, DataEntity const &prop);

std::ostream &operator<<(std::ostream &os, DataBase const &prop);

}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
