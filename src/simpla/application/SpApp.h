//
// Created by salmon on 17-4-13.
//

#ifndef SIMPLA_APPLICATION_H
#define SIMPLA_APPLICATION_H
//
#include <simpla/engine/TimeIntegrator.h>
#include <string>
#include "simpla/data/all.h"
namespace simpla {
namespace application {
struct SpApp : public data::Serializable, public data::EnableCreateFromDataTable<SpApp>, public data::Configurable {
    SP_OBJECT_BASE(SpApp);

   public:
    SpApp();
    virtual ~SpApp();
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable> t);
    virtual void Initialize();
    virtual void SetUp();
    virtual void Run();
    virtual void TearDown();
    virtual void Finalize();

    static std::shared_ptr<SpApp> Create(int argc, char **argv);

    void SetSchedule(std::shared_ptr<engine::TimeIntegrator> s) { m_schedule_ = s; };
    std::shared_ptr<engine::TimeIntegrator> GetSchedule() const { return m_schedule_; };

   private:
    std::shared_ptr<engine::TimeIntegrator> m_schedule_;
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
