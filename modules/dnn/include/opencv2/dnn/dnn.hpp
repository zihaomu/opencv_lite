/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_DNN_DNN_HPP
#define OPENCV_DNN_DNN_HPP

#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/core/async.hpp"

#include "../dnn/version.hpp"

#include <opencv2/dnn/dict.hpp>
#include "opencv2/core/utils/logger.hpp"

namespace cv {
namespace dnn {

CV__DNN_INLINE_NS_BEGIN
//! @addtogroup dnn
//! @{

    typedef std::vector<int> MatShape;

    /**
     * @brief Enum of computation precision supported by model.
     * @see Net::setPreferablePrecision
     */
    enum Precision
    {
        DNN_PRECISION_NORMAL = 0,
        DNN_PRECISION_LOW = 1,
        DNN_PRECISION_HIGH = 2,
    };

    /**
     * @brief Enum of computation backends supported by layers.
     * @see Net::setPreferableBackend
     */
    enum Backend
    {
        DNN_BACKEND_OPENCV = -1, // deprecated
        DNN_BACKEND_DEFAULT = 0, // CPU
        DNN_BACKEND_OPENCL, // For OpenCL GPU
        DNN_BACKEND_CUDA,   // For Nvidia GPU
    };

    //! deprecated. Currently, we only use Backend as target Device.
    enum Target
    {
        DNN_TARGET_CPU = 0,
        DNN_TARGET_OPENCL_GPU,
        DNN_TARGET_CUDA,
    };

    CV_EXPORTS std::vector< std::pair<Backend, Target> > getAvailableBackends();
    CV_EXPORTS_W std::vector<Target> getAvailableTargets(dnn::Backend be);

    /** @deprecated */
    CV_EXPORTS void enableModelDiagnostics(bool isDiagnosticsMode);

    /** @brief Deprecated.This class provides all data needed to initialize layer.
     *
     * It includes dictionary with scalar params (which can be read by using Dict interface),
     * blob params #blobs and optional meta information: #name and #type of layer instance.
    */
    class CV_EXPORTS LayerParams : public Dict
    {
    public:
        //TODO: Add ability to name blob params
        std::vector<Mat> blobs; //!< List of learned parameters stored as blobs.

        String name; //!< Name of the layer instance (optional, can be used internal purposes).
        String type; //!< Type name which was used for creating layer by layer factory (optional).
    };

    /** @brief This class allows to create and manipulate comprehensive artificial neural networks.
     *
     * Neural network is presented as directed acyclic graph (DAG), where vertices are Layer instances,
     * and edges specify relationships between layers inputs and outputs.
     *
     * Each network layer has unique integer id and unique string name inside its network.
     * LayerId can store either layer name or layer id.
     *
     * This class supports reference counting of its instances, i. e. copies point to the same instance.
     */
    class CV_EXPORTS_W_SIMPLE Net
    {
    public:

        CV_WRAP Net();  //!< Default constructor.
        CV_WRAP ~Net(); //!< Destructor frees the net only if there aren't references to the net anymore.

        /** Returns true if there are no layers in the network. */
        CV_WRAP bool empty() const;

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputName name for layer which output is needed to get
         *  @return blob for first output of specified layer.
         *  @details By default runs forward pass for the whole network.
         */
        CV_WRAP Mat forward(const String& outputName = String());

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputBlobs contains all output blobs for specified layer.
         *  @param outputName name for layer which output is needed to get
         *  @details If @p outputName is empty, runs forward pass for the whole network.
         */
        CV_WRAP void forward(OutputArrayOfArrays outputBlobs, const String& outputName = String());

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains blobs for first outputs of specified layers.
         *  @param outBlobNames names for layers which outputs are needed to get
         */
        CV_WRAP void forward(OutputArrayOfArrays outputBlobs,
                             const std::vector<String>& outBlobNames);

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains all output blobs for each layer specified in @p outBlobNames.
         *  @param outBlobNames names for layers which outputs are needed to get
         */
//        CV_WRAP_AS(forwardAndRetrieve) void forward(CV_OUT std::vector<std::vector<Mat> >& outputBlobs,
//                                                    const std::vector<String>& outBlobNames);

        /** @brief Load model params to this net.
         *  @param modelPath model path.
         */
        CV_WRAP void readNet(const String& model);

        /** @brief Load model params from the in-memory buffer to this net.
         *  @param framework Explicit framework name tag to determine a format.
         *  The following are supported name tags:
         *  "onnx", need ONNXRuntime backend.
         *  "mnn", need MNN backend.
         *  "trt", need TensorRT backend.
         *  @param buffer buffer memory address of the first byte of the buffer.
         *  @param sizeBuffer size of the buffer.
         */
        CV_WRAP void readNet(const char* framework, const char* buffer, size_t sizeBuffer);

        /**
         * @brief Ask network to use specific computation backend where it supported.
         * @param[in] backendId backend identifier.
         * @see Backend
         */
        CV_WRAP void setPreferableBackend(int backendId)
        {
            // TODO! Add OpenCL and CUDA supported.
            CV_LOG_WARNING(NULL, "setPreferableBackend do nothing. Currently only supports the CPU backend, and will support the CUDA backend in the future!");
            // do nothing.
        }

        /** @deprecated use setPreferableBackend is enough!*/
        CV_WRAP void setPreferableTarget(int targetId)
        {
            CV_LOG_WARNING(NULL, "setPreferableTarget is deprecated on opencv lite version, use setPreferableBackend to trigger target device!");
            // do nothing
        }

        /**
         * @brief Ask network to use specific computation precision model.
         * @param[in] precisionId precision identifier.
         * @see Precision
         */
        CV_WRAP void setPreferablePrecision(int precisionId);

        /**
         * @brief Ask network to forward model with specific number of threads.
         * @param[in] numThread number of threads.
         */
        CV_WRAP void setNumThreads(int numThread);

        /** @brief Sets the new input value for the network
         *  @param blob        A new blob. Should have CV_32F or CV_8U depth.
         *  @param name        A name of input layer.
         *  @param scalefactor An optional normalization scale.
         *  @param mean        An optional mean subtraction values.
         *  @see connect(String, String) to know format of the descriptor.
         *
         *  If scale or mean values are specified, a final input blob is computed
         *  as:
         * \f[input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\f]
         */
        CV_WRAP void setInput(InputArray blob, const String& name = "");

        CV_WRAP std::vector<std::string> getInputName();
        CV_WRAP std::vector<std::string> getOutputName();

        CV_WRAP std::vector<MatShape> getInputShape();
        CV_WRAP std::vector<MatShape> getOutputShape();

        /** @brief Enables or disables layer fusion in the network.
         * @param fusion true to enable the fusion, false to disable. The fusion is enabled by default.
         */
//        CV_WRAP void enableFusion(bool fusion);

        /** @brief Enables or disables the Winograd compute branch. The Winograd compute branch can speed up
         * 3x3 Convolution at a small loss of accuracy.
        * @param useWinograd true to enable the Winograd compute branch. The default is true.
        */
//        CV_WRAP void enableWinograd(bool useWinograd);

        class Impl;
        inline Impl* getImpl() const { return impl.get(); }
        inline Impl& getImplRef() const { CV_DbgAssert(impl); return *impl.get(); }
    protected:
        Ptr<Impl> impl;
    };

     /**
      * @brief Read deep learning network represented in one of the supported formats.
      * @param[in] model Binary file contains trained weights. The following file
      *                  extensions are expected for models from different frameworks:
      *                  * `*.onnx` (ONNX, https://onnx.ai/)
      * @param[in] config Text file contains network configuration. It could be a
      *                   file with the following extensions:
      *                  * `*.prototxt` (Caffe, http://caffe.berkeleyvision.org/)
      *                  * `*.pbtxt` (TensorFlow, https://www.tensorflow.org/)
      *                  * `*.cfg` (Darknet, https://pjreddie.com/darknet/)
      *                  * `*.xml` (DLDT, https://software.intel.com/openvino-toolkit)
      * @param[in] framework Explicit framework name tag to determine a format.
      * @returns Net object.
      *
      * This function automatically detects an origin framework of trained model
      * and calls an appropriate function such @ref readNetFromCaffe, @ref readNetFromTensorflow,
      * @ref readNetFromTorch or @ref readNetFromDarknet. An order of @p model and @p config
      * arguments does not matter.
      */
     CV_EXPORTS_W Net readNet(const String& model);

//     /**
//      * @brief Read deep learning network represented in one of the supported formats.
//      * @details This is an overloaded member function, provided for convenience.
//      *          It differs from the above function only in what argument(s) it accepts.
//      * @param[in] framework    Name of origin framework.
//      * @param[in] bufferModel  A buffer with a content of binary file with weights
//      * @param[in] bufferConfig A buffer with a content of text file contains network configuration.
//      * @returns Net object.
//      */
//     CV_EXPORTS_W Net readNet(const String& framework, const std::vector<uchar>& bufferModel,
//                              const std::vector<uchar>& bufferConfig = std::vector<uchar>());

    /** @brief Reads a network model <a href="https://onnx.ai/">ONNX</a>.
     *  @param onnxFile path to the .onnx file with text description of the network architecture.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     */
    CV_EXPORTS_W Net readNetFromONNX(const String &onnxFile);

    /** @brief Reads a network model from <a href="https://onnx.ai/">ONNX</a>
     *         in-memory buffer.
     *  @param buffer memory address of the first byte of the buffer.
     *  @param sizeBuffer size of the buffer.
     *  @returns Network object that ready to do forward, throw an exception
     *        in failure cases.
     *        // TODO, current it's not supported, need add supported
     */
    CV_EXPORTS Net readNetFromONNX(const char* buffer, size_t sizeBuffer);

    /** @brief Reads a network model from <a href="https://onnx.ai/">ONNX</a>
     *         in-memory buffer.
     *  @param buffer in-memory buffer that stores the ONNX model bytes.
     *  @returns Network object that ready to do forward, throw an exception
     *        in failure cases.
     *        // TODO, current it's not supported, need add supported
     */
    CV_EXPORTS_W Net readNetFromONNX(const std::vector<uchar>& buffer);

    /** @brief Reads a network model .trt file.
     *  @param trtFile path to the .trt file with text description of the network architecture.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     */
    CV_EXPORTS_W Net readNetFromTRT(const String &trtFile);

    /** @brief Reads a network model from .mnn in-memory buffer.
     *  @param buffer memory address of the first byte of the buffer.
     *  @param sizeBuffer size of the buffer.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     */
    CV_EXPORTS_W Net readNetFromMNN(const char* buffer, size_t sizeBuffer);

    /** @brief Reads a network model from .mnn file.
     *  @param mnnFile path to the .mnn file with text description of the network architecture.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     */
    CV_EXPORTS_W Net readNetFromMNN(const String &mnnFile);

    /** @brief Creates blob from .pb file.
     *  @param path to the .pb file with input tensor.
     *  @returns Mat.
     */
    CV_EXPORTS_W Mat readTensorFromONNX(const String& path);

    /** @brief Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,
     *  subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels.
     *  @param image input image (with 1-, 3- or 4-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p image values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @param crop flag which indicates whether image will be cropped after resize or not
     *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
     *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
     *  @returns 4-dimensional Mat with NCHW dimensions order.
     */
    CV_EXPORTS_W Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                   int ddepth=CV_32F);

    /** @brief Creates 4-dimensional blob from image.
     *  @details This is an overloaded member function, provided for convenience.
     *           It differs from the above function only in what argument(s) it accepts.
     */
    CV_EXPORTS void blobFromImage(InputArray image, OutputArray blob, double scalefactor=1.0,
                                  const Size& size = Size(), const Scalar& mean = Scalar(),
                                  bool swapRB=false, bool crop=false, int ddepth=CV_32F);


    /** @brief Creates 4-dimensional blob from series of images. Optionally resizes and
     *  crops @p images from center, subtract @p mean values, scales values by @p scalefactor,
     *  swap Blue and Red channels.
     *  @param images input images (all with 1-, 3- or 4-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p images values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @param crop flag which indicates whether image will be cropped after resize or not
     *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
     *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
     *  @returns 4-dimensional Mat with NCHW dimensions order.
     */
    CV_EXPORTS_W Mat blobFromImages(InputArrayOfArrays images, double scalefactor=1.0,
                                    Size size = Size(), const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                    int ddepth=CV_32F);

    /** @brief Creates 4-dimensional blob from series of images.
     *  @details This is an overloaded member function, provided for convenience.
     *           It differs from the above function only in what argument(s) it accepts.
     */
    CV_EXPORTS void blobFromImages(InputArrayOfArrays images, OutputArray blob,
                                   double scalefactor=1.0, Size size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                   int ddepth=CV_32F);

    /** @brief Parse a 4D blob and output the images it contains as 2D arrays through a simpler data structure
     *  (std::vector<cv::Mat>).
     *  @param[in] blob_ 4 dimensional array (images, channels, height, width) in floating point precision (CV_32F) from
     *  which you would like to extract the images.
     *  @param[out] images_ array of 2D Mat containing the images extracted from the blob in floating point precision
     *  (CV_32F). They are non normalized neither mean added. The number of returned images equals the first dimension
     *  of the blob (batch size). Every image has a number of channels equals to the second dimension of the blob (depth).
     */
    CV_EXPORTS_W void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_);

    /** @brief Print the Mat shape.
     * @param[in] blob mutil-dimensional array 1D, 2D, ... data in any data type.
     */
    CV_EXPORTS_W void printMatShape(InputArray blob);

    /** @brief Print the Mat source data pixcel by pixcel.
     * @param[in] blob mutil-dimensional array 1D, 2D, ... data in any data type.
     * @param[in] printLen By default, the first 100 elements of Mat are printed. Setting -1 prints all elements
     */
    CV_EXPORTS_W void printMatData(InputArray blob, int printLen = 100);

//! @}
CV__DNN_INLINE_NS_END
}
}

#include <opencv2/dnn/dnn.inl.hpp>

#endif  /* OPENCV_DNN_DNN_HPP */
