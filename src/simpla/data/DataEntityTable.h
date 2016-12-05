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

    virtual DataEntity &at(std::string const &key);

    virtual DataEntity const &at(std::string const &key) const;

    virtual void foreach(std::function<void(std::string const &key, DataEntity const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &key, DataEntity &)> const &fun);

    template<typename T> bool equal(std::string const &k, T const &v) const { return has(k) && at(k).equal(v); };

    virtual void insert(std::string const &key, std::shared_ptr<DataEntity> const &v);

    template<typename U>
    void insert(std::string const &key, U const &v) { insert(key, create_data_entity(v)); }

    struct WriteAccessor
    {
    public:
        WriteAccessor(DataEntityTable *t, std::string const &key) : m_table_(t), m_key_(key) {}

        ~WriteAccessor() {}

        template<typename U> DataEntity &operator=(U const &v) { m_table_->insert(m_key_, v); };

        template<typename U> U &operator U() { return as<U>(); };

        template<typename U> U &as() { return m_table_->at(m_key_).as<U>(); };
    private:
        DataEntityTable *m_table_;
        std::string m_key_;
    };


    struct ReadAccessor
    {
    public:
        ReadAccessor(DataEntityTable const *t, std::string const &key) : m_table_(t), m_key_(key) {}

        ~ReadAccessor() {}

        template<typename U> U const &operator U() const { return as<U>(); };

        template<typename U> U const &as() const { return m_table_->at(m_key_).as<U>(); };
    private:
        DataEntityTable const *m_table_;
        std::string m_key_;
    };

    WriteAccessor get(std::string const &key) { return WriteAccessor(this, key); };

    ReadAccessor get(std::string const &key) const { return ReadAccessor(this, key); };

    WriteAccessor operator[](std::string const &key) { return get(key); };

    ReadAccessor operator[](std::string const &key) const { return get(key); };





protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
