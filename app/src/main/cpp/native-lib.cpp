/*
 * Luca Anzalone
 */

#include <jni.h>

#include <string>
#include <vector>
#include <mutex>

#include <android/log.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>  // FAST
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/generic_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/dnn.h>
#define LOG_TAG "native-lib"
#define LOGD(...) \
  ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

#define JNI_METHOD(NAME) \
    Java_com_dev_anzalone_luca_facelandmarks_Native_##NAME

#define KERNEL_SIZE 5 // 3, 5, 7, 9

#define NV21 17
#define YV12 842094169
#define YUV_420_888 35
#define PYRAMIDS 3
#define MAX_FRAME_COUNT 5

using namespace std;
using namespace dlib;
// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

dlib::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

// global variables:
dlib::shape_predictor  predictor;
dlib::matrix<float,0,1> descriptor;

std::vector<dlib::matrix<float,0,1>> face_descriptors;
anet_type face_net;
std::mutex _mutex;
int imageFormat = NV21;

// -------------------------------------------------------------------------------------------------
// -- Lucas-Kanade Optical Flow Tracker
// -------------------------------------------------------------------------------------------------
namespace LK {
    // variables
    int frameCount = 0;
    bool isTracking = false;
    cv::Mat prev_img;
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> next_pts;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 25, 0.01);
    cv::Size ROI(20, 20);

    /** Initialize tracking with the current frame and detected landmarks */
    void start(cv::Mat &mat, dlib::full_object_detection &pts) {
        // release stuff..
        prev_img.release();
        prev_img = mat;
        prev_pts.clear();
        next_pts.clear();

        // consider the new points
        for (unsigned long i = 0; i < pts.num_parts(); i++) {
            auto pt = pts.part(i);
            prev_pts.push_back(cv::Point2f(pt.x(), pt.y()));
        }

        // reset count
        frameCount = 0;
        isTracking = true;
    }

    /** tracking points in the next captured frame */
    std::vector<cv::Point2f> track(cv::Mat &frame) {
        std::vector<uchar> status;
        std::vector<float> err;
        std::vector<cv::Point2f> tracked;

        // get the new points from the old one
        calcOpticalFlowPyrLK(prev_img, frame, prev_pts, next_pts, status, err,
                             ROI, PYRAMIDS, criteria);

        for (int i = 0; i < status.size(); ++i) {
            if (status[i] == 0) {
                // flow not found: take the old point
                tracked.push_back(prev_pts[i]);
            } else {
                // flow found: take the new point
                tracked.push_back(next_pts[i]);
            }
        }

        // switch the previous points and image with the current
        swap(prev_img, frame);
        swap(prev_pts, tracked);
        next_pts.clear();

        // increase tracking frame count
        if (frameCount++ > MAX_FRAME_COUNT) {
            isTracking = false;
        }

        return prev_pts;
    }
}
// -------------------------------------------------------------------------------------------------

extern "C"
JNIEXPORT void JNICALL
JNI_METHOD(loadModel)(JNIEnv* env, jclass, jstring detectorPath, jstring netPath) {
    try {
        const char *path = env->GetStringUTFChars(detectorPath, JNI_FALSE);
        const char *net = env->GetStringUTFChars(netPath, JNI_FALSE);
        _mutex.lock();
            // cause the later initialization of the tracking
            LK::isTracking = false;

            // load the shape predictor
            dlib::deserialize(path) >> predictor;
            dlib::deserialize(net) >> face_net;
        _mutex.unlock();

        env->ReleaseStringUTFChars(detectorPath, path); //free mem
        env->ReleaseStringUTFChars(netPath, net); //free mem
        LOGD("JNI: model loaded");

    } catch (dlib::serialization_error &e) {
        LOGD("JNI: failed to model -> %s", e.what());
    }
}
//--------------------------------------------------------------------------------------------------

void rotateMat(cv::Mat &mat, int rotation) {
    if (rotation == 90) { // portrait
        LOGD("JNI: rotation 90");
        cv::transpose(mat, mat);
        cv::flip(mat, mat, -1);
    } else if (rotation == 0) { // landscape-left
        LOGD("JNI: rotation 0");
        cv::flip(mat, mat, 1);
    } else if (rotation == 180) { // landscape-right
        LOGD("JNI: rotation 180");
        cv::flip(mat, mat, 0);
    }
}

extern "C"
JNIEXPORT void JNICALL
JNI_METHOD(setImageFormat)(JNIEnv* env, jclass, jint format) {
    imageFormat = format;
}

extern "C"
JNIEXPORT void JNICALL
JNI_METHOD(loadFace) (JNIEnv * env, jobject obj, jobjectArray myArray)
{
  int len1 = 16;
  jfloatArray dim =  (jfloatArray)env->GetObjectArrayElement(myArray, 0);
  int len2 = 128;



  for(int i=0; i<len1; ++i){
     jfloatArray oneDim = (jfloatArray)env->GetObjectArrayElement(myArray, i);
     jfloat *element = env->GetFloatArrayElements(oneDim, 0);
     dlib::matrix<float,0,1> fdata;
     fdata.set_size(128, 1);

     //allocate localArray[i] using len2
     for(int j=0; j<len2; ++j) {
        LOGD("____________ %f", element[j]);
        fdata(j,0) =  element[j];
     }
     face_descriptors.push_back(fdata);

     env->ReleaseFloatArrayElements( oneDim , (jfloat *)element, 0);
  }
  //TODO play with localArray, don't forget to release memory ;)
}



//--------------------------------------------------------------------------------------------------
//-- LANDMARK DETECTION
//--------------------------------------------------------------------------------------------------
extern "C"
JNIEXPORT jlongArray JNICALL
JNI_METHOD(detectLandmarks)(JNIEnv* env, jclass, jbyteArray yuvFrame, jint rotation, jint width, jint height, jint left, jint top, jint right, jint bottom) {
    LOGD("JNI: detectLandmarks");

    // copy content of frame into image
    jbyte *data = env->GetByteArrayElements(yuvFrame, 0);

    // convert yuv-frame to cv::Mat
    cv::Mat yuvMat(height + height / 2, width, CV_8UC1, (unsigned char *) data);
    cv::Mat grayMat(height, width, CV_8UC1);

    // to grayscale
    if (imageFormat == NV21)
        cv::cvtColor(yuvMat, grayMat, cv::COLOR_YUV2GRAY_NV21);  // was CV_YUV2GRAY_NV21
    else if (imageFormat == YV12)
        cv::cvtColor(yuvMat, grayMat, cv::COLOR_YUV2GRAY_YV12);  // was CV_YUV2GRAY_YV12

    // adjust rotation according to phone orientation
    rotateMat(grayMat, rotation);

    // crop face for enhancements
    cv::Rect faceROI(left, top, right - left, bottom - top);
    cv::Mat face = grayMat(faceROI);

    // apply filters
    cv::medianBlur(face, face, KERNEL_SIZE);  // remove noise
    cv::equalizeHist(face, face);  // improve contrast

    if (!LK::isTracking) {
        // -- DETECT LANDMARKS -- //

        // cv::mat to dlib::image
        dlib::cv_image<unsigned char> image(grayMat);

        // detect landmark points
        _mutex.lock();
        dlib::rectangle region(left, top, right, bottom);
        dlib::full_object_detection points = predictor(image, region);
        std::vector<dlib::matrix<rgb_pixel>> faces;
        dlib::matrix<rgb_pixel> face_chip;
        dlib::extract_image_chip(image, get_face_chip_details(points,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        std::vector<dlib::matrix<float,0,1>> descriptors = face_net(faces);


        for(int i = 0; i < 5; i++){
            LOGD("____________ %f __________ %f" , face_descriptors[5](i,0) , descriptors[0](i,0));
        }


        if (dlib::length(face_descriptors[5] - descriptors[0]) < 0.6){
            LOGD("____________ FACE RECOGNITION");
        }
        _mutex.unlock();

        // result
        auto num_points = points.num_parts();
        jsize len = (jsize) (num_points * sizeof(short)); // num_points * 2

        jlong buffer[len];
        jlongArray result = env->NewLongArray(len);

        // copy points in the buffer
        auto k = 0;
        for (unsigned long i = 0l; i < num_points; ++i) {
            dlib::point p = points.part(i);
            buffer[k++] = p.x();
            buffer[k++] = p.y();
        }

        // set the content of buffer into result array
        env->SetLongArrayRegion(result, 0, len, buffer);

        // free mem
        env->ReleaseByteArrayElements(yuvFrame, data, 0);

        // uncomment to enable tracking for the next frames
//        LK::start(grayMat, points);

        return result;

    } else {
        // -- COMPUTE LK-OPTICAL FLOW --
        auto trackedPts = LK::track(grayMat);

        // result
        auto num_points = trackedPts.size();
        jsize len = (jsize) (num_points * sizeof(short)); // num_points * 2

        jlong buffer[len];
        jlongArray result = env->NewLongArray(len);

        // copy tracked points in the buffer
        auto k = 0;
        for (unsigned long i = 0l; i < num_points; ++i) {
            auto p = trackedPts[i];
            buffer[k++] = static_cast<jlong>(p.x);
            buffer[k++] = static_cast<jlong>(p.y);
        }

        // set the content of buffer into result array
        env->SetLongArrayRegion(result, 0, len, buffer);

        // free mem
        env->ReleaseByteArrayElements(yuvFrame, data, 0);

        return result;
    }
}

//--------------------------------------------------------------------------------------------------