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

    virtual bool empty() const;

    virtual bool has(std::string const &key) const;

    virtual void foreach(std::function<void(std::string const &key, DataEntity const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &key, DataEntity &)> const &fun);

    template<typename T> bool check(std::string const &url, T const &v) const
    {
        DataEntity const *p = find(url);
        return p != nullptr && p->equal(v);
    };


    virtual DataEntity const *find(std::string const &url) const;

    virtual void parse(std::string const &str);

    template<typename U>
    virtual void parse(std::pair<std::string, U> const &k_v) { set_value(k_v.first, k_v.second); };

    template<typename T0, typename ...Args>
    virtual void parse(T0 const &a0, Args &&...args)
    {
        parse(a0);
        parse(std::forward<Args>(args)...);
    };

    /**
     *  set entity value to '''url'''
     * @param url
     * @return Returns a reference to the shared pointer of  the  modified entity, create parent '''table''' as needed.
     */

    virtual std::shared_ptr<DataEntity> &set(std::string const &key, std::shared_ptr<DataEntity> const &v);

    template<typename U> std::shared_ptr<DataEntity> &
    set_value(std::string const &url, U const &v) { return set(url, create_data_entity(v)); }

    virtual DataEntityTable *create_table(std::string const &url);

    /**
     *
     * @param url
     * @return Returns a reference to the shared pointer of  the entity with '''url'''.
     *      If no such entity exists, create a light entity, create parent table as needed.
     */
    virtual std::shared_ptr<DataEntity> &get(std::string const &url);


    template<typename U> U const &get_value(std::string const &url) const { return at(url).as<U>(); }

    template<typename U> U const &get_value(std::string const &url, U const &u) const
    {
        auto const *p = find(url);
        return p == nullptr ? u : p->as<U>();
    }


    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    virtual DataEntity &at(std::string const &key);

    virtual DataEntity const &at(std::string const &key) const;


    DataEntityLight &as_light(std::string const &url) { return at(url).as_light(); };

    DataEntityLight const &as_light(std::string const &url) const { return at(url).as_light(); };

    DataEntityHeavy &as_heavy(std::string const &url) { return at(url).as_heavy(); };

    DataEntityHeavy const &as_heavy(std::string const &url) const { return at(url).as_heavy(); };

    DataEntityTable &as_table(std::string const &url) { return at(url).as_table(); };

    DataEntityTable const &as_table(std::string const &url) const { return at(url).as_table(); };


protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
