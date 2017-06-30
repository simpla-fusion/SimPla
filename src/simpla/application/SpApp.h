//
// Created by salmon on 17-4-13.
//

#ifndef SIMPLA_APPLICATION_H
#define SIMPLA_APPLICATION_H
//
#include <string>
#include "simpla/data/all.h"
#include "simpla/utilities/SPObject.h"
#include "simpla/engine/Schedule.h"
namespace simpla {
namespace application {
struct SpApp : public SPObject, public data::Serializable {
    SP_OBJECT_HEAD(SpApp, SPObject);

   public:
    explicit SpApp(std::string const &s_name = "SpApp");
    virtual ~SpApp();
    SP_DEFAULT_CONSTRUCT(SpApp);

    using data::Serializable::Serialize;
    using data::Serializable::Deserialize;

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &cfg) override;

    void Initialize() override;
    void Update() override;
    virtual void Run();
    void TearDown() override;
    void Finalize() override;

    void SetContext(std::shared_ptr<engine::Context> s);
    std::shared_ptr<engine::Context> GetContext() const;

    void SetSchedule(std::shared_ptr<engine::Schedule> s);
    std::shared_ptr<engine::Schedule> GetSchedule() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace application{

#define SP_APP(_app_name, _app_desc)                                                                   \
    struct _APPLICATION_##_app_name : public application::SpApp {                                      \
        typedef _APPLICATION_##_app_name this_type;                                                    \
        static bool is_registered;                                                                     \
        _APPLICATION_##_app_name() {}                                                                  \
        _APPLICATION_##_app_name(this_type const &) = delete;                                          \
        virtual ~_APPLICATION_##_app_name() {}                                                         \
        void Deserialize(std::shared_ptr<data::DataTable>);                                            \
    };                                                                                                 \
    bool _APPLICATION_##_app_name::is_registered =                                                     \
        application::SpApp::RegisterCreator<_APPLICATION_##_app_name>(__STRING(_app_name), _app_desc); \
    void _APPLICATION_##_app_name::Deserialize(std::shared_ptr<data::DataTable> options)

#define SP_REGISITER_APP(_app_name, _app_desc)      \
    bool _APPLICATION_##_app_name##_is_registered = \
        application::SpApp::RegisterCreator<_app_name>(__STRING(_app_name), _app_desc);

}  // namespace simpla{
#endif  // SIMPLA_APPLICATION_H
