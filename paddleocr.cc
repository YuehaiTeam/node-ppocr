#include <napi.h>
#include <node_api.h>

#include <mutex>
#include <string>
#include <exception>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "src/ocr_det.h"
#include "src/ocr_rec.h"

mutex m_mutex;

class OCRInstance
{
public:
    PaddleOCR::DBDetector *m_det;
    PaddleOCR::CRNNRecognizer *m_rec;
    OCRInstance(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();
        std::string det_model_dir = info[0].As<Napi::String>();
        std::string rec_model_dir = info[1].As<Napi::String>();
        std::string char_list_file = info[2].As<Napi::String>();
        Napi::Object options = info[3].As<Napi::Object>();
        bool use_mkldnn = options.Get("use_mkldnn").As<Napi::Boolean>();
        bool use_tensorrt = options.Get("use_tensorrt").As<Napi::Boolean>();
        bool use_fp16 = options.Get("use_fp16").As<Napi::Boolean>();
        bool use_gpu = options.Get("use_gpu").As<Napi::Boolean>();
        int gpu_id = options.Get("gpu_id").As<Napi::Number>().Int32Value();
        int gpu_mem = options.Get("gpu_mem").As<Napi::Number>().Int32Value();
        int max_side_len = options.Get("max_side_len").As<Napi::Number>().Int32Value();
        int cpu_math_library_num_threads = options.Get("cpu_math_library_num_threads").As<Napi::Number>().Int32Value();
        float det_db_unclip_ratio = options.Get("det_db_unclip_ratio").As<Napi::Number>().FloatValue();
        float det_db_box_thresh = options.Get("det_db_box_thresh").As<Napi::Number>().FloatValue();
        float det_db_thresh = options.Get("det_db_thresh").As<Napi::Number>().FloatValue();

        m_det = new PaddleOCR::DBDetector(det_model_dir, use_gpu, gpu_id,
                                          gpu_mem, cpu_math_library_num_threads,
                                          use_mkldnn, max_side_len, det_db_thresh,
                                          det_db_box_thresh, det_db_unclip_ratio,
                                          false, use_tensorrt, use_fp16);

        m_rec = new PaddleOCR::CRNNRecognizer(rec_model_dir, use_gpu, gpu_id,
                                              gpu_mem, cpu_math_library_num_threads,
                                              use_mkldnn, char_list_file,
                                              use_tensorrt, use_fp16);
    }
    ~OCRInstance()
    {
        delete m_det;
        delete m_rec;
    }
};
static void addon_unload(napi_env env, void *data, void *hint)
{
    OCRInstance *ref = (OCRInstance *)data;
    delete ref;
}
Napi::Value load(const Napi::CallbackInfo &info)
{
    OCRInstance *inst = NULL;
    Napi::Env env = info.Env();
    try
    {
        m_mutex.lock();
        inst = new OCRInstance(info);
        m_mutex.unlock();
    }
    catch (std::exception &e)
    {
        throw Napi::Error::New(env, e.what());
    }
    catch (...)
    {
        exception_ptr ep = current_exception();
        throw Napi::Error::New(env, "Unknown CPP Error");
    }
    napi_set_instance_data(env, inst, addon_unload, NULL);
    return Napi::Boolean::New(env, true);
}

Napi::Value detect(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    OCRInstance *inst = NULL;
    napi_get_instance_data(env, (void **)&inst);
    if (inst == NULL)
    {
        return Napi::Boolean::New(env, false);
    }

    int width = info[0].As<Napi::Number>().Int32Value();
    int height = info[1].As<Napi::Number>().Int32Value();
    Napi::Buffer<uchar> buffer = info[2].As<Napi::Buffer<uchar>>();

    int bufferLength = buffer.Length();
    uchar *image = (uchar *)malloc(sizeof(uchar) * bufferLength);
    for (int i = 0; i < bufferLength; i += 4)
    {
        image[i + 0] = buffer[i + 0];
        image[i + 1] = buffer[i + 1];
        image[i + 2] = buffer[i + 2];
        image[i + 3] = buffer[i + 3];
    }
    cv::Mat srcimg = cv::Mat(height, width, CV_8UC4, image);
    cv::cvtColor(srcimg, srcimg, cv::COLOR_RGBA2RGB);

    vector<vector<vector<int>>> boxes;
    cv::Mat img_vis = inst->m_det->Run(srcimg, boxes);

    srcimg.release();
    img_vis.release();
    free(image);

    Napi::Array boxArr = Napi::Array::New(env, boxes.size());
    int i = 0;
    for (auto ii : boxes)
    {
        Napi::Object tmpObj = Napi::Object::New(env);
        Napi::Array tmpArr = Napi::Array::New(env, ii.size());
        int j = 0;
        for (auto jj : ii)
        {
            Napi::Array tmpArr2 = Napi::Array::New(env, jj.size());
            int k = 0;
            for (auto kk : jj)
            {
                tmpArr2[k++] = kk;
            }
            tmpArr[j++] = tmpArr2;
        }
        tmpObj.Set("box", tmpArr);
        boxArr[i++] = tmpObj;
    }
    return boxArr;
}

Napi::Value recognize(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    OCRInstance *inst = NULL;
    napi_get_instance_data(env, (void **)&inst);
    if (inst == NULL)
    {
        return Napi::Boolean::New(env, false);
    }

    int width = info[0].As<Napi::Number>().Int32Value();
    int height = info[1].As<Napi::Number>().Int32Value();
    Napi::Buffer<uchar> buffer = info[2].As<Napi::Buffer<uchar>>();

    int bufferLength = buffer.Length();
    uchar *image = (uchar *)malloc(sizeof(uchar) * bufferLength);
    for (int i = 0; i < bufferLength; i += 4)
    {
        image[i + 0] = buffer[i + 0];
        image[i + 1] = buffer[i + 1];
        image[i + 2] = buffer[i + 2];
        image[i + 3] = buffer[i + 3];
    }
    cv::Mat srcimg = cv::Mat(height, width, CV_8UC4, image);
    cv::cvtColor(srcimg, srcimg, cv::COLOR_RGBA2RGB);

    vector<std::string> res = inst->m_rec->RunOnly(srcimg);

    srcimg.release();
    free(image);

    Napi::Array boxArr = Napi::Array::New(env, res.size());
    int i = 0;
    for (auto ii : res)
    {
        boxArr[i++] = Napi::String::New(env, ii);
    }
    return boxArr;
}

Napi::Value ocr(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    OCRInstance *inst = NULL;
    napi_get_instance_data(env, (void **)&inst);
    if (inst == NULL)
    {
        return Napi::Boolean::New(env, false);
    }

    int width = info[0].As<Napi::Number>().Int32Value();
    int height = info[1].As<Napi::Number>().Int32Value();
    Napi::Buffer<uchar> buffer = info[2].As<Napi::Buffer<uchar>>();

    int bufferLength = buffer.Length();
    uchar *image = (uchar *)malloc(sizeof(uchar) * bufferLength);
    for (int i = 0; i < bufferLength; i += 4)
    {
        image[i + 0] = buffer[i + 0];
        image[i + 1] = buffer[i + 1];
        image[i + 2] = buffer[i + 2];
        image[i + 3] = buffer[i + 3];
    }
    cv::Mat srcimg = cv::Mat(height, width, CV_8UC4, image);
    cv::cvtColor(srcimg, srcimg, cv::COLOR_RGBA2RGB);

    vector<vector<vector<int>>> boxes;
    cv::Mat img_vis = inst->m_det->Run(srcimg, boxes);
    vector<std::string> res = inst->m_rec->Run(boxes, srcimg);

    srcimg.release();
    img_vis.release();
    free(image);

    Napi::Array boxArr = Napi::Array::New(env, boxes.size());
    int i = 0;
    for (auto ii : boxes)
    {
        Napi::Object tmpObj = Napi::Object::New(env);
        Napi::Array tmpArr = Napi::Array::New(env, ii.size());
        int j = 0;
        for (auto jj : ii)
        {
            Napi::Array tmpArr2 = Napi::Array::New(env, jj.size());
            int k = 0;
            for (auto kk : jj)
            {
                tmpArr2[k++] = kk;
            }
            tmpArr[j++] = tmpArr2;
        }
        tmpObj.Set("box", tmpArr);
        tmpObj.Set("text", Napi::String::New(env, res[i * 2]));
        tmpObj.Set("confidence", Napi::String::New(env, res[i * 2 + 1]));
        boxArr[i++] = tmpObj;
    }
    return boxArr;
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(Napi::String::New(env, "ocr"), Napi::Function::New(env, ocr));
    exports.Set(Napi::String::New(env, "detect"), Napi::Function::New(env, detect));
    exports.Set(Napi::String::New(env, "recognize"), Napi::Function::New(env, recognize));
    exports.Set(Napi::String::New(env, "load"), Napi::Function::New(env, load));
    return exports;
}

NODE_API_MODULE(addon, Init)
