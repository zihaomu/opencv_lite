// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "backend.hpp"
#include "factory.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// reserve this class for ONNXRuntime, tflite.
class BackendRegistry
{
public:
    typedef std::vector< std::pair<Backend, Target> > BackendsList;
    const BackendsList & getBackends() const { return backends; }
    static BackendRegistry & getRegistry()
    {
        static BackendRegistry impl;
        return impl;
    }

private:
    BackendRegistry()
    {
        backends.push_back(std::make_pair(DNN_BACKEND_OPENCV, DNN_TARGET_CPU));
    }

    BackendsList backends;
};


std::vector<std::pair<Backend, Target>> getAvailableBackends()
{
    return BackendRegistry::getRegistry().getBackends();
}

std::vector<Target> getAvailableTargets(Backend be)
{
    if (be == DNN_BACKEND_DEFAULT)
        be = (Backend)getParam_DNN_BACKEND_DEFAULT();

    std::vector<Target> result;
    const BackendRegistry::BackendsList all_backends = getAvailableBackends();
    for (BackendRegistry::BackendsList::const_iterator i = all_backends.begin(); i != all_backends.end(); ++i)
    {
        if (i->first == be)
            result.push_back(i->second);
    }
    return result;
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
