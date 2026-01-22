# OpenCV-lite

一个轻量版的opencv，专注dnn模型部署场景，并融入mediapipe主要功能，方便用户在各个平台移植。
本仓库做了哪些工作：
1. 裁剪opencv，扩展数据类型，增加fp16， int64， bool类型；
2. 删除opencv dnn自有推理引擎，对接第三方推理引擎，现在已支持：
    1. onnxruntime，CPU和CUDA GPU
    2. MNN，CPU
    3. tflite，CPU和安卓GPU
 3. 全套的mediapipe 移植。mediapipe是很棒的项目。

逐一回答为什么要实现以上几点：
1. 为什么要裁剪opencv？
在我从事ai模型部署以来，opencv在这个领域模块的使用频率有限，但是多余的模块会极大的增加编译时间。有些模块引入了下载链接，这也给网络有问题的用户带来了困扰。

1. 为什么要删除原有dnn模块，而引入第三方推理引擎？
过去三年时间，所接触到的部署平台包括：服务端，pc端和移动端。我发现没有一个推理引擎能够在所有平台所有情况获得最优解。
MNN作为通用推理框架，基本能够做到在：服务端，pc端和移动端上均衡的性能，不过有个大问题，就是需要将模型格式进行转换，有些模型格式无法转换，就嗝屁了。开发mnn算子的工程量太大，需要了解不少mnn库的知识。
这时候，加入onnxruntime的支持，也是一个很好的补充，windows，linux，mac都有预编译好的库，用起来很方便。
为什么需要tflite？在安卓平台上，使用一些比较宽的模型时，使用gpu推理是必须的，目前测试了，只有tflite的opencl速度是最好的，tflite的cpu是被mnn按在地上摩擦的。

tflite，onnxruntime以及mnn的调用api都不算太复杂，不过没学过的同学还是需要一些学习成本的，为了统一一下调用接口，这里将他们映射到了opencv dnn的api中。

3. 为什么需要移植一套mediapipe？
mediapipe是很棒的项目，但是官方目前的没有想让这个库编程跨平台的想法，而想把这个库做成安卓独占的存在，其他平台的用户想要用mediapipe，是很困难的。mediapipe也算是比较复杂的能力了，正好作为本仓库的一个附加模块。增加一些测试样例。



## OpenCV: Open Source Computer Vision Library


### Resources

* Homepage: <https://opencv.org>
  * Courses: <https://opencv.org/courses>
* Docs: <https://docs.opencv.org/4.x/>
* Q&A forum: <https://forum.opencv.org>
  * previous forum (read only): <http://answers.opencv.org>
* Issue tracking: <https://github.com/opencv/opencv/issues>
* Additional OpenCV functionality: <https://github.com/opencv/opencv_contrib>
* Donate to OpenCV: <https://opencv.org/support/>


### Contributing

Please read the [contribution guidelines](https://github.com/opencv/opencv/wiki/How_to_contribute) before starting work on a pull request.

#### Summary of the guidelines:

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Clean up "oops" commits before submitting;
* Follow the [coding style guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide).

### Additional Resources

* [Submit your OpenCV-based project](https://form.jotform.com/233105358823151) for inclusion in Community Friday on opencv.org
* [Subscribe to the OpenCV YouTube Channel](http://youtube.com/@opencvofficial) featuring OpenCV Live, an hour-long streaming show
* [Follow OpenCV on LinkedIn](http://linkedin.com/company/opencv/) for daily posts showing the state-of-the-art in computer vision & AI
* [Apply to be an OpenCV Volunteer](https://form.jotform.com/232745316792159) to help organize events and online campaigns as well as amplify them
* [Follow OpenCV on Mastodon](http://mastodon.social/@opencv) in the Fediverse
* [Follow OpenCV on Twitter](https://twitter.com/opencvlive)
* [OpenCV.ai](https://opencv.ai): Computer Vision and AI development services from the OpenCV team.
