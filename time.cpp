#include "timer.h"

Time::Time() : first(true)
{
}

void Time::clear() {
	this->first = true;
	this->startIdentifiers.clear();
	this->beginTimes.clear();
	this->endTimes.clear();
	this->stackIdentifiers = std::stack<std::string>();
}

void Time::begin(const std::string &msg) {
	if (this->first) {
		this->shipTimer.start();
		this->first = false;
 	}
	this->stackIdentifiers.push(msg);
	this->beginTimes[msg] = this->shipTimer.get_elapsed_time_mcs();
}

void Time::end(const std::string &msg) {
	check(this->stackIdentifiers.size() != 0 && this->stackIdentifiers.top() == msg);
	startIdentifiers.push_back(msg);
	this->endTimes[msg] = this->shipTimer.get_elapsed_time_mcs();
	this->stackIdentifiers.pop();
}

unsigned long long Time::getTimeMicroseconds(const std::string &msg) {
	for (int i = 0; i < static_cast <int> (this->startIdentifiers.size()); i++) {
		if (this->startIdentifiers[i] == msg) {
			return this->endTimes[msg] - this->beginTimes[msg];
		}
	}
	throw std::string("no times are there: " + msg);
}

unsigned int Time::getTimeMilliseconds(const std::string &msg) {
	return (unsigned int)(getTimeMicroseconds(msg) / 1e3);
}

float Time::getTimeMillisecondsFloat(const std::string &msg) {
	return getTimeMicroseconds(msg) / static_cast<float>(1e3);
}

unsigned int Time::getTimeSec(const std::string &msg) {
	return (unsigned int)(getTimeMicroseconds(msg) / 1e6);
}

float Time::getTimeSecFloat(const std::string &msg) {
	return getTimeMicroseconds(msg) / static_cast<float>(1e6);
}
