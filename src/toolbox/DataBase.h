//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DICT_H
#define SIMPLA_DICT_H


#include "SIMPLA_config.h"

#include <typeinfo>
#include <string>
#include <memory>
#include <ostream>
#include <iomanip>

namespace simpla { namespace toolbox
{

struct DataType;

struct DataSpace;

class DataBase
{
public:


    DataBase() {};

    virtual  ~DataBase() {};

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataBase); }

    virtual bool eval(std::string path) { return true; };

    virtual bool open(std::string path, int flag = 0) { return true; };

    virtual void close() {};

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual size_t size() const =0;

    virtual bool empty() const =0;

    virtual bool has(std::string const &key) const =0;

    struct Entity
    {
    public:
        Entity() {}

        virtual ~Entity() {}

        virtual const std::type_info &type() const =0;

        virtual std::ostream &print(std::ostream &os, int indent = 0) const =0;

        virtual bool is_null() const =0;

//        virtual DataType data_type() =0;
//
//        virtual DataSpace data_space()  =0;

        virtual const void *data() const =0;

        virtual void *data() =0;
    };

    virtual Entity const &value() const =0;

    virtual Entity &value() =0;

    /**
     *  as container
     */
    virtual void set(std::string const &, std::shared_ptr<DataBase> const &) =0;

    /**
    *  if key exists then return ptr else create and return ptr
    * @param key
    * @return
    */
    virtual std::shared_ptr<DataBase> get(std::string const &key)=0;

    /**
     *  if key exists then return ptr else return null
     * @param key
     * @return
     */
    virtual std::shared_ptr<DataBase> at(std::string const &key)=0;

    virtual std::shared_ptr<const DataBase> at(std::string const &key) const =0;

    /**
     *   for_each(<lambda>)  lets container decide the traversing algorithm,
     *   which is better than iterator to traverse containers.
     *   *. easy to overload, need  not to implement iterator class
     *   *. easy to parallism
     *
     */

    virtual void for_each(std::function<void(std::string const &, DataBase &)> const &fun)=0;

    virtual void for_each(std::function<void(std::string const &, DataBase const &)> const &fun) const =0;


};

std::ostream &operator<<(std::ostream &os, DataBase const &prop);

}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
