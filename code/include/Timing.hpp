#ifndef __TIMING_CLASS__
#define __TIMING_CLASS__ 

#include <chrono>

class Timing
{
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime, endTime;

public:
	Timing();

	void start();
	void stop();
	double elapsed();

};

Timing::Timing()
{

}

void Timing::start()
{
	startTime = std::chrono::high_resolution_clock::now();
}
void Timing::stop()
{
	endTime = std::chrono::high_resolution_clock::now();
}

double Timing::elapsed()
{
	auto e = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	return(e.count());
}


#endif