//
// Created by salmon on 7/8/15.
//

#ifndef SIMPLA_GTL_DESIGN_PATTERN_OBSERVER_H
#define SIMPLA_GTL_DESIGN_PATTERN_OBSERVER_H

#include <memory>
#include <map>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

namespace simpla
{

struct Observable;

struct Observer
{
	typedef boost::uuids::uuid id_type;

	const id_type id;

	Observer() :
			id(boost::uuids::random_generator()())
	{
	}

	virtual ~Observer()
	{
		if (m_subject_ != nullptr)
		{
			m_subject_->remove_observer(*this);
		}
	};

	void set_subject(Observable &subject)
	{
		m_subject_ = subject.shared_from_this();
	}

	void unset_subject()
	{
		m_subject_ = nullptr;
	}

	template<typename ...Args>
	void notify(Args &&...args)
	{

	}

	virtual void notify() = 0;

private:
	std::shared_ptr<Observable> m_subject_;

};


struct Observable : public std::enable_shared_from_this<Observable>
{
	typedef typename Observer::id_type id_type;

	std::map<id_type, std::shared_ptr<Observer>> m_observers_;


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
			item.second->notify(std::forward<Args>(args)...);
		}
	}


	void add_observer(std::shared_ptr<Observer> observer)
	{

		auto res = m_observers_.emplace(std::make_pair(observer->id, observer));

		if (res.second)
		{
			res.first->second->set_subject(*this);
		}

	};

	template<typename T, typename ...Args>
	typename std::enable_if<std::is_polymorphic<Observer>::value,
			std::shared_ptr<T>>::type create_observer(Args &&...args)
	{
		auto res = std::make_shared<T>(std::forward<Args>(args)...);

		add_observer(std::dynamic_pointer_cast<Observable>(res));

		return res;

	};


	void remove_observer(Observer &observer)
	{
		auto it = m_observers_.find(observer.id);

		if (it != m_observers_.end())
		{
			it->second->unset_subject();
			m_observers_.erase(it);
		}
	}

	void remove_observer(std::shared_ptr<Observer> observer)
	{
		remove_observer(*observer);
	}


};


}// namespace simpla
#endif //SIMPLA_GTL_DESIGN_PATTERN_OBSERVER_H
