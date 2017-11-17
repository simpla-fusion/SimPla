//
// Created by salmon on 17-8-30.
//

#ifndef SIMPLA_DATANODEEXT_H
#define SIMPLA_DATANODEEXT_H

namespace simpla {
namespace data {

class DataEntryEntity : public DataEntry {
    SP_DEFINE_FANCY_TYPE_NAME(DataEntryEntity, DataEntry)
   protected:
    DataEntryEntity() : DataEntry(DN_ENTITY) {}

   public:
    ~DataEntryEntity() override = default;
    size_type size() const override { return 1; }

    std::shared_ptr<DataEntity> GetEntity() const override = 0;
    size_type SetEntity(std::shared_ptr<DataEntity> const&) override = 0;
};
#define SP_DATA_NODE_ENTITY_HEAD(_CLASS_NAME_)              \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataEntryEntity) \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                         \
   protected:                                               \
    _CLASS_NAME_();                                         \
                                                            \
   public:                                                  \
    ~_CLASS_NAME_();                                        \
    std::shared_ptr<DataEntity> GetEntity() const override; \
    size_type SetEntity(std::shared_ptr<DataEntity> const& entity) override;

class DataEntryArray : public DataEntry {
    SP_DEFINE_FANCY_TYPE_NAME(DataEntryArray, DataEntry)

   protected:
    DataEntryArray() : DataEntry(DN_ARRAY) {}

   public:
    ~DataEntryArray() override = default;

    size_type size() const override = 0;

    std::shared_ptr<DataEntry> CreateNode(eNodeType) const override = 0;

    size_type Set(std::string const& s, std::shared_ptr<DataEntry> const& v) override {
        return Set(std::stoi(s, nullptr, 10), v);
    };
    size_type Add(std::string const& s, std::shared_ptr<DataEntry> const& v) override {
        return Add(std::stoi(s, nullptr, 10), v);
    };
    size_type Delete(std::string const& s) override { return Delete(std::stoi(s, nullptr, 10)); };
    std::shared_ptr<DataEntry> Get(std::string const& s) const override { return Get(std::stoi(s, nullptr, 10)); };

    size_type Set(size_type s, std::shared_ptr<DataEntry> const& v) override = 0;
    size_type Add(size_type s, std::shared_ptr<DataEntry> const& v) override = 0;
    size_type Delete(size_type s) override = 0;
    std::shared_ptr<DataEntry> Get(size_type s) const override = 0;
    size_type Add(std::shared_ptr<DataEntry> const& v) override = 0;
};
#define SP_DATA_NODE_ARRAY_HEAD(_CLASS_NAME_)                                \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataEntryArray)                   \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                                          \
   protected:                                                                \
    _CLASS_NAME_();                                                          \
                                                                             \
   public:                                                                   \
    ~_CLASS_NAME_();                                                         \
                                                                             \
    size_type size() const override;                                         \
                                                                             \
    std::shared_ptr<DataEntry> CreateNode(eNodeType) const override;          \
                                                                             \
    size_type Set(size_type s, std::shared_ptr<DataEntry> const& v) override; \
    size_type Add(size_type s, std::shared_ptr<DataEntry> const& v) override; \
    size_type Delete(size_type s) override;                                  \
    std::shared_ptr<DataEntry> Get(size_type s) const override;               \
    size_type Add(std::shared_ptr<DataEntry> const& v) override;

class DataEntryTable : public DataEntry {
    SP_DEFINE_FANCY_TYPE_NAME(DataEntryTable, DataEntry)

   protected:
    DataEntryTable() : DataEntry(DN_TABLE) {}

   public:
    ~DataEntryTable() override = default;

   public:
    size_type size() const override = 0;

    std::shared_ptr<DataEntry> CreateNode(eNodeType) const override = 0;

    size_type Set(std::string const& uri, std::shared_ptr<DataEntry> const& v) override = 0;
    size_type Add(std::string const& uri, std::shared_ptr<DataEntry> const& v) override = 0;
    size_type Delete(std::string const& uri) override = 0;
    std::shared_ptr<DataEntry> Get(std::string const& uri) const override = 0;
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataEntry>)> const& f) const override = 0;
};

#define SP_DATA_NODE_TABLE_HEAD(_CLASS_NAME_)                                           \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataEntryTable)                              \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                                                     \
   protected:                                                                           \
    _CLASS_NAME_();                                                                     \
                                                                                        \
   public:                                                                              \
    ~_CLASS_NAME_();                                                                    \
                                                                                        \
    size_type size() const override;                                                    \
    std::shared_ptr<DataEntry> CreateNode(eNodeType) const override;                     \
    size_type Set(std::string const& uri, std::shared_ptr<DataEntry> const& v) override; \
    size_type Add(std::string const& uri, std::shared_ptr<DataEntry> const& v) override; \
    size_type Delete(std::string const& uri) override;                                  \
    std::shared_ptr<DataEntry> Get(std::string const& uri) const override;               \
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataEntry>)> const& f) const override;

struct DataEntryFunction : public DataEntry {
    SP_DEFINE_FANCY_TYPE_NAME(DataEntryFunction, DataEntry)
   public:
    size_type size() const override { return 0; };
};

#define SP_DATA_ENTITY_HEAD_HEAD(_CLASS_NAME_)              \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataEntryFunction) \
    SP_DATA_NODE_HEAD(_CLASS_NAME_)                           \
   protected:                                                 \
    _CLASS_NAME_();                                           \
                                                              \
   public:                                                    \
    ~_CLASS_NAME_();
}
}
#endif  // SIMPLA_DATANODEEXT_H
