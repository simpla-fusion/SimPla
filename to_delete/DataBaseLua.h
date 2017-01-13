//
// Created by salmon on 16-10-7.
//

#ifndef SIMPLA_DATABASELUA_H
#define SIMPLA_DATABASELUA_H

#include "DataBase.h"

namespace simpla
{
namespace toolbox
{

class DataBaseLua : public DataBase
{
public:
    struct Entity;
    struct iterator;
    struct const_iterator;

    DataBaseLua() {};

    ~DataBaseLua() {};

    void swap(DataBaseLua &other);

    std::ostream &print(std::ostream &os, int indent = 0) const;

    bool open(std::string path);

    void close();

    /**
     * as value
     * @{
     */

    bool is_table() const;

    bool has_value() const;

    Entity const &value() const;

    Entity &value();

    /** @} */

    /**
     *  as container
     *  @{
     */

    size_t size() const;

    bool empty() const;

    typedef typename std::map<std::string, std::shared_ptr<DataBaseLua>>::
    iterator iterator;

    typedef typename std::map<std::string, std::shared_ptr<DataBaseLua>>::
    const_iterator const_iterator;

    bool has(std::string const &key) const;

    iterator find(std::string const &key);

    std::pair<iterator, bool> insert(std::string const &, std::shared_ptr<DataBaseLua> &);

    /**
    *  if key exists then return ptr else create and return ptr
    * @param key
    * @return
    */
    std::shared_ptr<DataBase> get(std::string const &key);

    /**
     *  if key exists then return ptr else return null
     * @param key
     * @return
     */
    std::shared_ptr<DataBase> at(std::string const &key);

    std::shared_ptr<const DataBase> at(std::string const &key) const;

    iterator begin();

    iterator end();

    const_iterator begin() const;

    const_iterator end() const;

    /** @}*/
};


class DataBaseLua::Entity : public DataBase::Entity
{
    Entity();

    ~Entity();

    void swap(Entity &other);

    bool empty() const;

    bool is_null() const;

    const void *data() const;

    void *data();

    DataType data_type();

    DataSpace data_space();
};


class DataBaseLua::iterator : public DataBase::iterator
{

};

class DataBaseLua::const_iterator : public DataBase::const_iterator
{

};


}
}
#endif //SIMPLA_DATABASELUA_H
