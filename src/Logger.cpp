/*
* Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "rtxmu/Logger.h"

// Logger function callback which allows client to implement logging
namespace rtxmu
{
    Level Logger::s_loggerVerbosity = Level::DISABLED;
    void (*Logger::s_loggerCallback)(const char*) = nullptr;
    void Logger::setLoggerSettings(Level verbosity)
    {
        s_loggerVerbosity = verbosity;
    }
    void Logger::setLoggerCallback(void (*loggerCallback)(const char*))
    {
        s_loggerCallback = loggerCallback;
    }
    void Logger::log(Level verbosity, const char* msg)
    {
        if (verbosity <= s_loggerVerbosity)
        {
            if (s_loggerCallback != nullptr)
            {
                s_loggerCallback(msg);
            }
        }
    }
    bool Logger::isEnabled(Level verbosity)
    {
        return (verbosity <= s_loggerVerbosity) ? true : false;
    }
}// end rtxmu namespace
