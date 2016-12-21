//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DATATREE_H_
#define SIMPLA_DATATREE_H_


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
/** @ingroup data */
/**
 * @brief  a @ref DataEntity tree, a key-value table of @ref DataEntity, which is similar as Group in HDF5, but all node/table are DataEntity.
 */
class DataTable : public DataEntity
{
SP_OBJECT_HEAD(DataTable, DataEntity);

public:

    DataTable();

    DataTable(DataTable const &) = delete;

    DataTable(DataTable &&) = delete;

    virtual  ~DataTable();

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

    virtual void parse() {};

    virtual void parse(std::string const &str);

    template<typename U>
    void parse(std::pair<std::string, U> const &k_v) { set_value(k_v.first, k_v.second); };

    template<typename T0, typename ...Args>
    void parse(T0 const &a0, Args &&...args)
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

    virtual DataTable *create_table(std::string const &url);

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

//    template<typename U> U const &get_value(std::string const &url, U const &u)
//    {
//        auto *p = find(url);
//
//        if (p != nullptr) { return p->as<U>(); } else { return set_value(url, u)->as<U>(); }
//    }

    /**
     *
     * @param key
     * @return Returns a reference to the mapped value of the element with key equivalent to key.
     *      If no such element exists, an exception of type std::out_of_range is thrown.
     */
    virtual DataEntity &at(std::string const &key);

    virtual DataEntity const &at(std::string const &key) const;


    LightData &as_light(std::string const &url) { return at(url).as_light(); };

    LightData const &as_light(std::string const &url) const { return at(url).as_light(); };

    HeavyData &as_heavy(std::string const &url) { return at(url).as_heavy(); };

    HeavyData const &as_heavy(std::string const &url) const { return at(url).as_heavy(); };

    DataTable &as_table(std::string const &url) { return at(url).as_table(); };

    DataTable const &as_table(std::string const &url) const { return at(url).as_table(); };


protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DATATREE_H_
