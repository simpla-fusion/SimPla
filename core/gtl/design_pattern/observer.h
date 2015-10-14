/**
 *  @file observer.h
 */

#ifndef SIMPLA_GTL_DESIGN_PATTERN_OBSERVER_H
#define SIMPLA_GTL_DESIGN_PATTERN_OBSERVER_H

#include <memory>
#include <set>

namespace simpla
{
namespace gtl {

template<typename SIGNATURE> struct Observable;
template<typename SIGNATURE> struct Observer;

template<typename ...Args>
struct Observer<void(Args...)>
{


	Observer()
	{
	}

	virtual ~Observer()
	{
		if (m_subject_ != nullptr)
		{
			m_subject_->disconnect(this);
		}
	};

	void connect(Observable &subject)
	{
		m_subject_ = subject.shared_from_this();
	}

	void disconnect()
	{
		std::shared_ptr<Observable>(nullptr).swap(m_subject_);
	}


	virtual void notify(Args ...) = 0;

private:
	std::shared_ptr<Observable> m_subject_;

};

template<typename Signature>
struct Observable<Signature> : public std::enable_shared_from_this<Observable<Signature>>
{
	typedef Observer<Signature> observer_type;

	std::set<std::shared_ptr<observer_type>> m_observers_;


	Observable()
	{
	}

	virtual ~Observable()
	{
	}

	template<typename ...Args>
	void notify(Args &&...args)
	{
		for (auto &item:m_observers_)
		{
			item->notify(std::forward<Args>(args)...);
		}
	}


	void connect(std::shared_ptr<observer_type> observer)
	{
		observer->connect(*this);
		m_observers_.insert(observer);
	};

	template<typename T, typename ...Args>
	typename std::enable_if<std::is_polymorphic<observer_type>::value,
			std::shared_ptr<T>>::type create_observer(Args &&...args)
	{
		auto res = std::make_shared<T>(std::forward<Args>(args)...);

		connect(std::dynamic_pointer_cast<observer_type>(res));

		return res;

	};


	void disconnect(observer_type *observer)
	{
		auto it = m_observers_.find(observer);

		if (it != m_observers_.end())
		{
			(**it).disconnect();

			m_observers_.erase(it);
		}
	}

	void remove_observer(std::shared_ptr<observer_type> &observer)
	{
		disconnect(observer.get());
	}


};


}}//  namespace simpla::gtl
#endif //SIMPLA_GTL_DESIGN_PATTERN_OBSERVER_H
