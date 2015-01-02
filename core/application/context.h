/**
 * \file context.h
 *
 * \date    2014年9月18日  上午9:33:53 
 * \author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

namespace simpla
{

class Context
{
public:
	virtual ~Context();

	virtual void setup(int argc, char const** argv);

	virtual void body();

	virtual void split();

	virtual void sync();
};
struct ContextList
{
	std::map<std::string, std::shared_ptr<Context>> list_;

	std::string add(std::string const & name,
			std::shared_ptr<Context> const & p);

	template<typename T>
	std::string add(std::string const & name, std::shared_ptr<T> const & p)
	{
		list_[name] = std::dynamic_pointer_cast<Context>(p);
		return "Context" + ToString(list_.size()) + "_" + name;
	}
	template<typename T>
	std::string add(std::string const & name)
	{
		return add(name, std::make_shared<T>());
	}

	std::ostream & print(std::ostream & os);

	void run();
	void sync();
	void setup(int argc, char const ** argv);

}
;

template<typename T>
std::string RegisterContext(std::string const &name)
{
	return SingletonHolder<ContextList>::instance().template add<T>(name);
}
inline void RunAllContext()
{
	SingletonHolder<ContextList>::instance().run();
}
inline void SetUpContext(int argc, char const ** argv)
{
	SingletonHolder<ContextList>::instance().setup(argc, argv);
}
#define CONTEXT(_context_name) SP_APP(_context_name,context,Context)

}  // namespace simpla

#endif /* CORE_APPLICATION_CONTEXT_H_ */
