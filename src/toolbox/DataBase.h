//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DICT_H
#define SIMPLA_DICT_H
//
// Created by salmon on 16-10-7.
//



#include <map>
#include <memory>
#include "../sp_config.h"
#include "DataType.h"
#include "DataSpace.h"

namespace simpla { namespace toolbox
{


class DataBase
{
public:
    struct Entity;
    struct iterator;
    struct const_iterator;

    DataBase() {};

    virtual  ~DataBase() {};

    virtual std::ostream &print(std::ostream &os, int indent = 0) const =0;

    virtual bool open(std::string path) =0;

    virtual void close() =0;

    virtual bool is_table() const =0;

    virtual bool has_value() const =0;

    virtual size_t size() const =0;

    virtual bool empty() const =0;

    virtual bool has(std::string const &key) const =0;

    virtual Entity const &value() const =0;

    virtual Entity &value() =0;

    virtual iterator find(std::string const &key)=0;

    virtual std::pair<iterator, bool> insert(std::string const &, std::shared_ptr<DataBase> &);

    /**
    *  if key exists then return ptr else create and return ptr
    * @param key
    * @return
    */
    virtual std::shared_ptr<DataBase> get(std::string const &key);

    /**
     *  if key exists then return ptr else return null
     * @param key
     * @return
     */
    virtual std::shared_ptr<DataBase> at(std::string const &key)=0;

    virtual std::shared_ptr<const DataBase> at(std::string const &key) const =0;

    virtual iterator DataBase::begin()=0;

    virtual iterator DataBase::end() =0;

    virtual const_iterator DataBase::begin() const =0;

    virtual const_iterator DataBase::end() const =0;


};

struct DataBase::Entity
{
public:
    Entity() {}

    virtual ~Entity() {}

    virtual void swap(Entity &other) =0;

    virtual const std::type_info &type() const =0;

    virtual bool is_null() const =0;

    virtual DataType data_type() =0;

    virtual DataSpace data_space()  =0;

    virtual const void *data() const =0;

    virtual void *data() =0;
};

struct DataBase::iterator
{

};
//
//struct DataFuction : public Entity
//{
//    /**
//      *  as function
//      *  @{
//      */
//protected:
//
//    virtual Entity pop_return();
//
//    virtual void push_parameter(Entity const &);
//
//private:
//    template<typename TFirst> inline void push_parameters(TFirst &&first) { push_parameter(Entity(first)); }
//
//    template<typename TFirst, typename ...Args> inline
//    void push_parameters(TFirst &&first, Args &&...args)
//    {
//        push_parameters(std::forward<TFirst>(first));
//        push_parameters(std::forward<Args>(args)...);
//    };
//public:
//
//    template<typename ...Args> inline
//    Entity call(Args &&...args)
//    {
//        push_parameters(std::forward<Args>(args)...);
//        return pop_return();
//    };
//
//    template<typename ...Args> inline
//    Entity operator()(Args &&...args) { return call(std::forward<Args>(args)...); };
//};
//struct DataBase::iterator
//{
//    iterator() {};
//
//    virtual ~iterator() {};
//
//    virtual bool is_equal(iterator const &other) const;
//
//    virtual std::pair<std::string, std::shared_ptr<DataBase>> get() const;
//
//    virtual iterator &next();
//
//    virtual std::pair<std::string, std::shared_ptr<DataBase>> value() const;
//
//    std::pair<std::string, std::shared_ptr<DataBase>> operator*() const { return value(); };
//
//    std::pair<std::string, std::shared_ptr<DataBase>> operator->() const { return value(); };
//
//    bool operator!=(iterator const &other) const { return !is_equal(other); };
//
//    iterator &operator++() { return next(); }
//};

std::ostream &operator<<(std::ostream &os, DataBase const &prop) { return prop.print(os, 0); }

}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
