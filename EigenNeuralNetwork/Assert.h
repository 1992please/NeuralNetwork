#pragma once

#include "stdio.h"


#define ASSERTIONS_ENABLED 1

#if ASSERTIONS_ENABLED
// define some inline assembly that causes a break
// into the debugger -- this will be different on each
// target CPU
static void reportAssertionFailure(const char* expr, const char* file, int line)
{
	printf("%s\n %s\n %d\n", expr, file, line);
}

// check the expression and fail if it is false
#define ASSERT(expr) \
	if (expr) { } \
	else \
	{ \
	reportAssertionFailure(#expr, \
	__FILE__, __LINE__); \
	  __debugbreak(); \
	}
#else
#define ASSERT(expr) // evaluates to nothing
#endif