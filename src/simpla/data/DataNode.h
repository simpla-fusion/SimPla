//
// Created by salmon on 17-8-18.
//

#ifndef SIMPLA_DATAENTRY_H
#define SIMPLA_DATAENTRY_H

#include <memory>
namespace simpla {
namespace data {
class DataEntity;
class DataNode : public std::enable_shared_from_this<DataNode> {
   public:
    DataNode() = default;
    virtual ~DataNode() = default;
    SP_DEFAULT_CONSTRUCT(DataNode);

    virtual size_type GetNumberOfChildren() const { return 0; }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri) { return nullptr; }
    virtual std::shared_ptr<DataNode> GetNode(std::string const& uri) const { return nullptr; }

    virtual std::shared_ptr<DataNode> Root() { return GetNode("/"); }
    virtual std::shared_ptr<DataNode> Parent() const { return GetNode(".."); }

    virtual std::shared_ptr<DataNode> Child() const { return nullptr; }
    virtual std::shared_ptr<DataNode> Next() const { return nullptr; }

    virtual std::string Key() const { return ""; }
    virtual std::shared_ptr<DataEntity> Value() const { return nullptr; }

    virtual int SetValue(std::shared_ptr<DataEntity> const& v) { return 0; }

    virtual std::shared_ptr<DataEntity> GetValueByName(std::string const& s) const { return nullptr; }
    virtual int SetValueByName(std::string const& k, std::shared_ptr<DataEntity> const& v) { return 0; }

    virtual std::shared_ptr<DataEntity> GetValueByIndex(index_type idx) const { return nullptr; }
    virtual int SetValueByIndex(index_type idx, std::shared_ptr<DataEntity> const& v) { return 0; }
    virtual int AddValue(std::shared_ptr<DataEntity> const& v) { return SetValueByIndex(-1, v); }
};
class DataNodeWithKey : public DataNode {
    std::string m_key_;
    std::shared_ptr<DataEntity> m_value_;

   public:
    explicit DataNodeWithKey(std::string k, std::shared_ptr<DataEntity> v = "")
        : m_key_(std::move(k)), m_value_(std::move(v)) {}
    ~DataNodeWithKey() override = default;
    SP_DEFAULT_CONSTRUCT(DataNodeWithKey);

    std::shared_ptr<DataEntity> Value() const override { return m_value_; }
    std::string Key() const override { return m_key_; }
};

class DataNodeWithOutKey : public DataNode {
    std::shared_ptr<DataEntity> m_value_;

   public:
    explicit DataNodeWithOutKey(std::shared_ptr<DataEntity> v) : m_value_(std::move(v)) {}
    ~DataNodeWithOutKey() override = default;
    SP_DEFAULT_CONSTRUCT(DataNodeWithOutKey);

    std::shared_ptr<DataEntity> Value() const override { return m_value_; }
};
std::ostream& operator<<(std::ostream, DataNode const&);
}  // namespace data
}  // namespace simpla

#endif  // SIMPLA_DATAENTRY_H
