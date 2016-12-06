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
#include <map>


#include "DataEntity.h"

namespace simpla { namespace data
{

class DataEntityTable : public DataEntity
{
    SP_OBJECT_HEAD(DataEntityTable, DataEntity);

public:

    DataEntityTable();

    DataEntityTable(DataEntityTable const &) = delete;

    DataEntityTable(DataEntityTable &&) = delete;

    virtual  ~DataEntityTable();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual bool is_table() const { return true; };

    virtual bool is_null() const { return empty(); };

    virtual bool empty() const;

    virtual bool has(std::string const &key) const;

    virtual bool check(std::string const &key);

    virtual void foreach(std::function<void(std::string const &key, DataEntity const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &key, DataEntity &)> const &fun);

    template<typename T> bool equal(std::string const &k, T const &v) const { return has(k) && at(k).equal(v); };

    virtual void set(std::string const &key, std::shared_ptr<DataEntity> const &v);


    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual std::shared_ptr<DataEntity> &get(std::string const &url);

    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    virtual DataEntity &at(std::string const &key);

    virtual DataEntity const &at(std::string const &key) const;


    template<typename U>
    void set_value(std::string const &key, U const &v) { set(key, create_data_entity(v)); }


    template<typename U> U const &get_value(std::string const &url) const { return at(url).as<U>(); }

    template<typename U> U const &get_value(std::string const &url, U const &u) const
    {
        try { return get_value<U>(url); } catch (...) { return u; }
    }

    DataEntityLight &get_light(std::string const &url) { return get(url)->as_light(); };

    DataEntityLight const &get_light(std::string const &url) const { return at(url).as_light(); };

    DataEntityHeavy &get_heavy(std::string const &url) { return get(url)->as_heavy(); };

    DataEntityHeavy const &get_heavy(std::string const &url) const { return at(url).as_heavy(); };

    DataEntityTable &get_table(std::string const &url) { return get(url)->as_table(); };

    DataEntityTable const &get_table(std::string const &url) const { return at(url).as_table(); };


protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
