#pragma once

#include <sys/time.h>

#include <ctime>

// clang-format off
/**
 * 0 -> None,      // Clock is not initialized
 * 1 -> CPU_TIME,  // Clock calculates time ranges using ctime and CLOCKS_PER_SEC 
 * 2 -> REAL_TIME, // Clock calculates time by asking the operating system how  much real time passed
 */
// clang-format on
long double take_time(const int mode = 2) {
  if (mode == 1) {
    // Use ctime
    return (long double)clock();
  } else if (mode == 2) {
    // Query operating system

    /* Linux, MacOS, ... */
    struct timeval tv;
    gettimeofday(&tv, NULL);

    long double measure = tv.tv_usec;
    measure /= 1000000.0;              // Convert to seconds
    measure += (long double)tv.tv_sec; // Add seconds part
    return measure;
  }

  // If mode == NONE, clock has not been initialized, then throw exception
  throw "Clock not initialized to a time taking mode!";
}
