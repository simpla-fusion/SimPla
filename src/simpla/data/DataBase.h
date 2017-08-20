//
// Created by salmon on 17-3-6.
//

#ifndef SIMPLA_DATABACKEND_H
#define SIMPLA_DATABACKEND_H

#include "simpla/SIMPLA_config.h"

#include <memory>
#include <regex>
#include <typeindex>
#include <typeinfo>
#include <vector>
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/ObjectHead.h"
#include "simpla/utilities/SingletonHolder.h"

namespace simpla {
namespace data {
class DataNode;
class DataEntity;
class DataBase : public Factory<DataBase>, public std::enable_shared_from_this<DataBase> {
    SP_DEFINE_FANCY_TYPE_NAME(DataBase, Factory<DataBase>);

   protected:
    DataBase() = default;

   public:
    ~DataBase() override = default;
    SP_DEFAULT_CONSTRUCT(DataBase)

    static std::shared_ptr<DataBase> New(std::string const& uri = "");

    virtual std::shared_ptr<DataNode> Root() = 0;

    virtual int Connect(std::string const& authority, std::string const& path, std::string const& query,
                        std::string const& fragment) = 0;
    virtual int Disconnect() = 0;
    virtual int Flush() = 0;

    /**
      * @brief Get the number of entities in this table
      * @return
      */
    virtual bool isNull() const = 0;

    /**
     * @brief Get entities that are selected by the '''uri''',
     * @return if nothing is selected return nullptr
     */
    virtual std::shared_ptr<DataNode> Get(std::string const& uri) const = 0;

    /**
     * @brief  put v to uri,
     * @return
     */
    virtual int Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) = 0;
    virtual int Set(std::string const& uri, const std::shared_ptr<DataNode>& v) = 0;

    /**
     * @brief  add v to uri,
     *          if uri does not exist then create an array
     *          else if uri is not an array then throw runtim error
     * @return
     */
    virtual int Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) = 0;
    /**
     * @brief  delete entities selected by uri
     */
    virtual int Delete(std::string const& uri) = 0;

    static int s_num_of_pre_registered_;

};  // class DataBase {
#define SP_DATABASE_DECLARE_MEMBERS(_CLASS_NAME_)                                                \
    SP_DEFINE_FANCY_TYPE_NAME(_CLASS_NAME_, DataBase)                                                       \
   protected:                                                                                    \
    _CLASS_NAME_();                                                                              \
                                                                                                 \
   public:                                                                                       \
    ~_CLASS_NAME_() override;                                                                    \
    SP_DEFAULT_CONSTRUCT(_CLASS_NAME_);                                                          \
    template <typename... Args>                                                                  \
    static std::shared_ptr<_CLASS_NAME_> New(Args&&... args) {                                   \
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));           \
    };                                                                                           \
                                                                                                 \
   private:                                                                                      \
    struct pimpl_s;                                                                              \
    pimpl_s* m_pimpl_;                                                                           \
                                                                                                 \
   public:                                                                                       \
    int Connect(std::string const& authority, std::string const& path, std::string const& query, \
                std::string const& fragment) override;                                           \
    int Disconnect() override;                                                                   \
    bool isNull() const override;                                                                \
    int Flush() override;                                                                        \
    std::shared_ptr<DataNode> Get(std::string const& uri) const override;                        \
    int Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) override;              \
    int Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) override;              \
    int Delete(std::string const& uri) override;                                                 \
                                                                                                 \
   public:
// class DataBackendFactory : public design_pattern::Factory<std::string, DataBase>, public concept::Printable {
//    typedef design_pattern::Factory<std::string, DataBase> base_type;
//    SP_OBJECT_BASE(DataBackendFactory);
//
//   public:
//    DataBackendFactory();
//    ~DataBackendFactory() override;
//    SP_DEFAULT_CONSTRUCT(DataBackendFactory)
//    std::vector<std::string> GetBackendList() const;
//    std::shared_ptr<DataBase> Create(std::string const& uri, std::string const& ext_param = "");
//
//    void RegisterDefault();
//};
//#define GLOBAL_DATA_BACKEND_FACTORY SingletonHolder<DataBackendFactory>::instance()
}  // namespace data {
}  // namespace simpla{
#endif  // SIMPLA_DATABACKEND_H
