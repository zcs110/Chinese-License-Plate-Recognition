//
//  Yolov5.cpp
//  ChinesePlate
//
//  Created by Nature on 2023/12/28.
//  Copyright © 2023 lprSample. All rights reserved.
//

#include "Yolov5.h"

#include <ncnn/ncnn/layer.h>
#include <ncnn/ncnn/net.h>
//#include "benchmark.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ncnn/ncnn/cpu.h>
#include <opencv2/dnn/dnn.hpp>
#include <future>
#include <iostream>

using namespace cv;
using namespace std;

// crnn使用
std::vector<std::string> plate_chars = { "#","京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "学", "警", "港", "澳", "挂", "使", "领", "民", "航",
    "危",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "险", "品"};

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }
    
    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);
    
    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    
    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;
        
        while (faceobjects[j].prob < p)
            j--;
        
        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);
            
            i++;
            j--;
        }
    }
    
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;
    
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    
    const int n = faceobjects.size();
    
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }
    
    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];
        
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            
            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        
        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    
    /***************************************************************
     *  @brief     函数作用：未知
     *
     const ncnn::Mat& anchors：即自定义设置的锚框,
     int stride：步长如8、16、32
     const ncnn::Mat& in_pad：输入的mat
     const ncnn::Mat& feat_blob：输出的mat
     float prob_threshold：未知
     std::vector<Object>& objects：处理结果存放的vector
     *  @note      备注
     *  @Sample usage:     函数的使用方法
     **************************************************************/
    
    const int num_grid = feat_blob.h;  //获取输出mat的h值，此处为3840=48*80,此处考虑的是stride为8的值
    
    int num_grid_x;
    int num_grid_y;
    
    //    按照stride缩放，由宽和高相对大小对决定对基于w还是h缩放，最后将缩放后的w和h赋值给x和y
    //    这也是为什么输入一定会处理为32的倍数的原因
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;  // 这里是w更小，输入的w为384，h为640，故num_grid_y=640/8=80，num_grid_x=48*80/80=48
        num_grid_x = num_grid / num_grid_y;
    }
    //    cout<<"num_grid_x："<<num_grid_x<<endl;
    //    cout<<"num_grid_y："<<num_grid_y<<endl;
    const int num_class = feat_blob.w - 13;  // 这里w等于14，减去前面四个xywh，以及conf还有四个点的8个坐标一共13个，剩下的就是类别数
    const int num_anchors = anchors.w / 2;  // anchors的数量等于anchors的w来除以2，这里的anchors的w为6，则num_anchors为3
    //  torch.Size([1, 3, 80, 48, 14]),stride为8时的结果，经过conv之后的结果
    for (int q = 0; q < num_anchors; q++)  // 遍历3，即torch.Size([1, 3, 80, 48, 14])中的第2维度
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        
        const ncnn::Mat feat = feat_blob.channel(q);  // 获取某个stride的三个channel之一
        
        for (int i = 0; i < num_grid_y; i++)  // 遍历80，即torch.Size([1, 3, 80, 48, 14])中的第三维度
        {
            for (int j = 0; j < num_grid_x; j++) // 遍历48，即torch.Size([1, 3, 80, 48, 14])中的第四维度
            {
                const float* featptr = feat.row(i * num_grid_x + j); // 对torch.Size([1, 3, 80, 48, 14])中的第四维度中的48遍历获取其值，其值应该是一个数组，包含14个数
                float box_confidence = sigmoid(featptr[4]);  // 将这个数组中的第四也就是实际第五个的conf进行sigmoid，变成0-1，赋值给锚框的置信度
                if (box_confidence >= prob_threshold)  // 判断置信度是否大于预设值，只有大于的才会进入到结果中
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + 8 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score);  // 整体置信度阈值，也就是锚框置信度*类别置信度
                    if (confidence >= prob_threshold)
                    {
                        
                        // 这里是只对xywh做了sigmoid，其中类别conf在上面已经做过了，即：float confidence = box_confidence * sigmoid(class_score);
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);
                        
                        float p1x = featptr[5];
                        float p1y = featptr[6];
                        float p2x = featptr[7];
                        float p2y = featptr[8];
                        float p3x = featptr[9];
                        float p3y = featptr[10];
                        float p4x = featptr[11];
                        float p4y = featptr[12];
                        
                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;
                        
                        
                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;
                        // # landmark的进一步处理
                        p1x = p1x * anchor_w + j * stride;
                        p1y = p1y * anchor_h + i * stride;
                        p2x = p2x * anchor_w + j * stride;
                        p2y = p2y * anchor_h + i * stride;
                        p3x = p3x * anchor_w + j * stride;
                        p3y = p3y * anchor_h + i * stride;
                        p4x = p4x * anchor_w + j * stride;
                        p4y = p4y * anchor_h + i * stride;
                        
                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;
                        
                        Object obj;
                        obj.x = x0;
                        obj.y = y0;
                        obj.w = x1 - x0;
                        obj.h = y1 - y0;
                        obj.label = "";
                        obj.color = "";
                        obj.prob = confidence;
                        obj.p1x = p1x;
                        obj.p1y = p1y;
                        obj.p2x = p2x;
                        obj.p2y = p2y;
                        obj.p3x = p3x;
                        obj.p3y = p3y;
                        obj.p4x = p4x;
                        obj.p4y = p4y;
                        
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}
Yolov5::Yolov5(){
    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);
}

Yolov5::~Yolov5(){
    
}
// 加载YOLO模型
int Yolov5::load(int _target_size, bool use_gpu)
{
    
    yolov5.clear();
    crnn.clear();
    color_net.clear();
    
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    target_size = _target_size;
    
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;
    
    // use vulkan compute
#if NCNN_VULKAN
    yolov5.opt.use_vulkan_compute = use_gpu;
#endif
    yolov5.opt = opt;
    
    // 加载模型参数和权重
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"best" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"best" ofType:@"bin"];
    int rp = yolov5.load_param([parmaPath UTF8String]);// 加载模型参数
    int rm = yolov5.load_model([binPath UTF8String]);  // 加载模型权重
    if (rp == 0 && rm == 0) {
        printf("best 模型加载成功!\n"); // 加载成功
    } else {
        fprintf(stderr, "best 模型加载失败, param:%d model:%d\n", rp, rm); // 加载失败
    }
    
    // 加载颜色模型参数和权重
    ncnn::Option opt_crnn;
    opt_crnn.lightmode = true;
    opt_crnn.num_threads = 4;
    opt_crnn.blob_allocator = &g_blob_pool_allocator;
    opt_crnn.workspace_allocator = &g_workspace_pool_allocator;
    opt_crnn.use_packing_layout = true;
#if NCNN_VULKAN
    opt_crnn.use_vulkan_compute = use_gpu;
#endif
    crnn.opt = opt_crnn;
    NSString *parmaPatht = [[NSBundle mainBundle] pathForResource:@"plate_rec_color" ofType:@"param"];
    NSString *binPatht = [[NSBundle mainBundle] pathForResource:@"plate_rec_color" ofType:@"bin"];
    int rpt = crnn.load_param([parmaPatht UTF8String]);// 加载模型参数
    int rmt = crnn.load_model([binPatht UTF8String]);  // 加载模型权重
    if (rpt == 0 && rmt == 0) {
        printf("plate_rec_color 模型加载成功!\n"); // 加载成功
    } else {
        fprintf(stderr, "plate_rec_color 模型加载失败, param:%d model:%d\n", rpt, rmt); // 加载失败
    }
    
    ncnn::Option opt_color;
    opt_color.lightmode = true;
    opt_color.num_threads = 4;
    opt_color.blob_allocator = &g_blob_pool_allocator;
    opt_color.workspace_allocator = &g_workspace_pool_allocator;
    opt_color.use_packing_layout = true;
#if NCNN_VULKAN
    opt_color.use_vulkan_compute = use_gpu;
#endif
    color_net.opt = opt_color;
    NSString *parmaPathts = [[NSBundle mainBundle] pathForResource:@"color-sim" ofType:@"param"];
    NSString *binPathts = [[NSBundle mainBundle] pathForResource:@"color-sim" ofType:@"bin"];
    int rpts = color_net.load_param([parmaPathts UTF8String]);// 加载模型参数
    int rmts = color_net.load_model([binPathts UTF8String]);  // 加载模型权重
    if (rpts == 0 && rmts == 0) {
        printf("color-sim 模型加载成功!\n"); // 加载成功
    } else {
        fprintf(stderr, "color-sim 模型加载失败, param:%d model:%d\n", rpt, rmt); // 加载失败
    }
    return 0;
}
cv::Mat UIImageToMats(UIImage *image) {
    cv::Mat outputMat;
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    size_t width = CGImageGetWidth(image.CGImage);
    size_t height = CGImageGetHeight(image.CGImage);
    size_t bytesPerRow = CGImageGetBytesPerRow(image.CGImage);
    
    if (colorSpace && width > 0 && height > 0) {
        if (CGColorSpaceGetModel(colorSpace) == kCGColorSpaceModelRGB) {
            // RGB图像
            CGDataProviderRef provider = CGImageGetDataProvider(image.CGImage);
            CFDataRef data = CGDataProviderCopyData(provider);
            const uint8_t *bytes = CFDataGetBytePtr(data);
            
            cv::Mat inputMat((int)height, (int)width, CV_8UC4, (void *)bytes, bytesPerRow);
            //车牌图片二值化// 转换颜色通道
            cv::cvtColor(inputMat, outputMat, cv::COLOR_BGRA2BGR);
            CFRelease(data);
            //            printf("支持的颜色空间+++++++++");
        } else {
            printf("不支持的颜色空间");
        }
    }
    return outputMat;
}
UIImage* MatToUIImage(cv::Mat& mat) {
    // 检查输入的 mat 是否为空
    if (mat.empty()) {
        return nil;
    }
    
    // 检查 mat 的通道数，只支持单通道和三通道图像
    if (mat.channels() != 1 && mat.channels() != 3) {
        return nil;
    }
    
    // 使用 NSData 创建一个临时的数据缓冲区
    NSData *data = [NSData dataWithBytes:mat.data length:mat.total() * mat.elemSize()];
    
    // 获取图像的宽度和高度
    int width = mat.cols;
    int height = mat.rows;
    int bytesPerRow = (int)mat.step;
    
    // 使用位图上下文创建 UIImage
    CGColorSpaceRef colorSpace;
    if (mat.channels() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrderDefault;
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(width, height, 8, 8 * mat.elemSize(), bytesPerRow, colorSpace, bitmapInfo, provider, NULL, NO, kCGRenderingIntentDefault);
    
    UIImage *uiImage = [UIImage imageWithCGImage:imageRef];
    
    // 释放资源
    CGColorSpaceRelease(colorSpace);
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    //    printf("回转 Image ++++++++++++");
    
    return uiImage;
}
cv::Mat get_split_merge(const cv::Mat& img,cv::Mat& upperImage) {
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));
    upperImage = img_upper;
    return out;
}
// 目标检测
std::vector<Object> Yolov5::detect(UIImage *image,std::string &plateStr,std::string &plateColor, float prob_threshold, float nms_threshold)
{
    std::vector<Object> proposals;
    //to do image 转 rgb
    cv::Mat rgb = UIImageToMats(image);
    proposals = detectWithMat(rgb,plateStr,plateColor);
    return proposals;
}
std::vector<Object> Yolov5::detectWithMat( cv::Mat image,std::string &plateStr,std::string &plateColor,  float prob_threshold, float nms_threshold){
    std::vector<Object> objects;
    int width = image.cols;
    int height = image.rows;
    plateStr   = "";
    plateColor = "";
    // 调整大小以适应模型的输入大小
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat ina = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
    //这个操作确保了模型输入宽度是32的倍数，从而能够获得更好的计算性能，因为在某些硬件上，处理32的倍数的数据会更加高效。所以当 w 等于640时，wpad 的计算结果是负数，这意味着不需要填充，因为输入的宽度已经是32的倍数
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(ina, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = yolov5.create_extractor();
    //        ex.input("images", in_pad);
    ex.input("data", in_pad);
    ncnn::Mat mat_out1,mat_out2,mat_out3;
    bool ret1 = ex.extract("stride_8", mat_out1); //# stride 8
    bool ret2 = ex.extract("stride_16", mat_out2); //# stride 16
    bool ret3 = ex.extract("stride_32", mat_out3); //# stride 32
    if (ret1 || ret2 || ret3) {
        printf("有一个失败了");
    }else{
        printf("提取图层成功");
    }
    std::vector<Object> proposals;
    // stride 8
    {
//        ncnn::Mat out;
//        ex.extract("stride_8", out);
        ncnn::Mat anchors(6);
        anchors[0] = 4.f;
        anchors[1] = 5.f;
        anchors[2] = 8.f;
        anchors[3] = 10.f;
        anchors[4] = 13.f;
        anchors[5] = 16.f;
        
        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, mat_out1, prob_threshold, objects8);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }
    //
    // stride 16
    {
//        ncnn::Mat out;
//        //ex.extract("781", out);
//        ex.extract("stride_16", out);
        
        ncnn::Mat anchors(6);
        anchors[0] = 23.f;
        anchors[1] = 29.f;
        anchors[2] = 43.f;
        anchors[3] = 55.f;
        anchors[4] = 73.f;
        anchors[5] = 105.f;
        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, mat_out2, prob_threshold, objects16);
        
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    // stride 32
    {
//        ncnn::Mat out;
//        //ex.extract("801", out);
//        ex.extract("stride_32", out);
        ncnn::Mat anchors(6);
        anchors[0] = 146.f;
        anchors[1] = 217.f;
        anchors[2] = 231.f;
        anchors[3] = 300.f;
        anchors[4] = 335.f;
        anchors[5] = 433.f;
        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, mat_out3, prob_threshold, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    int count = (int)picked.size();
    
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
        
        // adjust offset to original unpadded
        float x0 = (objects[i].x - (wpad / 2)) / scale;
        float y0 = (objects[i].y - (hpad / 2)) / scale;
        
        float p1x = (objects[i].p1x - (wpad / 2)) / scale;
        float p1y = (objects[i].p1y - (hpad / 2)) / scale;
        float p2x = (objects[i].p2x - (wpad / 2)) / scale;
        float p2y = (objects[i].p2y - (hpad / 2)) / scale;
        float p3x = (objects[i].p3x - (wpad / 2)) / scale;
        float p3y = (objects[i].p3y - (hpad / 2)) / scale;
        float p4x = (objects[i].p4x - (wpad / 2)) / scale;
        float p4y = (objects[i].p4y - (hpad / 2)) / scale;
        
        float x1 = (objects[i].x + objects[i].w- (wpad / 2)) / scale;
        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;
        
        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);
        
        p1x = std::max(std::min(p1x, (float) (width - 1)), 0.f);
        p1y = std::max(std::min(p1y, (float) (height - 1)), 0.f);
        p2x = std::max(std::min(p2x, (float) (width - 1)), 0.f);
        p2y = std::max(std::min(p2y, (float) (height - 1)), 0.f);
        p3x = std::max(std::min(p3x, (float) (width - 1)), 0.f);
        p3y = std::max(std::min(p3y, (float) (height - 1)), 0.f);
        p4x = std::max(std::min(p4x, (float) (width - 1)), 0.f);
        p4y = std::max(std::min(p4y, (float) (height - 1)), 0.f);
        
        objects[i].x = x0;
        objects[i].y = y0;
        
        objects[i].w= x1 - x0;
        objects[i].h = y1 - y0;
        
        objects[i].p1x = p1x;
        objects[i].p1y = p1y;
        objects[i].p2x = p2x;
        objects[i].p2y = p2y;
        objects[i].p3x = p3x;
        objects[i].p3y = p3y;
        objects[i].p4x = p4x;
        objects[i].p4y = p4y;
    }
    
    //OCR
    std::vector<Object> proResult;
    for (size_t i=0; i<objects.size(); i++)
    {
        // letterbox pad to multiple of 32
        Object obj = objects[i];

        float new_x1 = obj.p3x - obj.x;
        float new_y1 = obj.p3y - obj.y;
        float new_x2 = obj.p4x - obj.x;
        float new_y2 = obj.p4y - obj.y;
        float new_x3 = obj.p2x - obj.x;
        float new_y3 = obj.p2y - obj.y;
        float new_x4 = obj.p1x - obj.x;
        float new_y4 = obj.p1y - obj.y;
        
        cv::Point2f src_points[4];
        cv::Point2f dst_points[4];
        
        //通过Image Watch查看的二维码四个角点坐标
        src_points[0]=cv::Point2f(new_x1, new_y1);
        src_points[1]=cv::Point2f(new_x2, new_y2);
        src_points[2]=cv::Point2f(new_x3, new_y3);
        src_points[3]=cv::Point2f(new_x4, new_y4);
        //期望透视变换后二维码四个角点的坐标
        dst_points[0]=cv::Point2f(0.0, 0.0);
        dst_points[1]=cv::Point2f(168.0, 0.0);
        dst_points[2]=cv::Point2f(0.0, 48.0);
        dst_points[3]=cv::Point2f(168.0, 48.0);
   
        cv::Mat rotation,rotation1,img_warp,tempCV;
        cv::Rect_<float> rect;
        rect.x = obj.x;
        rect.y = obj.y;
        rect.height = obj.h;
        rect.width = obj.w;

        cv::Mat upperImage;
        cv::Mat ROI = image(rect).clone();
        
//        tempCV = get_split_merge(ROI,upperImage);
//        执行拟合识别度下降了
//        rotation = getPerspectiveTransform(src_points,dst_points);
//        cout<< "\n" <<"width:"<<rect.width << " height:" <<rect.height << endl;
//        warpPerspective(ROI,ROI,rotation,cv::Size(168, 48));
        
        vector<string> plate_color_result(2);
        plate_color_result = crnn_rec(ROI);

        string plate_str=plate_color_result[0];
        plateStr = plate_str;
        //        string color_names[3] = {
        ////                "blue", "green","yellow"
        //                "蓝", "绿", "黄"
        //        };
        //        int color_code = color_rec_1(ROI);
        //        string color_name = color_names[color_code];
        string color_name = plate_color_result[1];
        plateColor = color_name;
    }
    return proResult;
}

#pragma mark - recognition
std::vector<std::string> Yolov5::crnn_rec(const cv::Mat& bgr){
    cv::Mat img = bgr;
    //获取图片的宽
    int w = img.cols;
    //获取图片的高
    int h = img.rows;
    
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 168, 48);
    float mean[3] = { 149.94, 149.94, 149.94 };
    float norm[3] = { 0.020319,0.020319,0.020319 };
    //对图片进行归一化,将像素归一化到-1~1之间
    in.substract_mean_normalize(mean, norm);
    
    ncnn::Extractor ex = crnn.create_extractor();
    ex.set_light_mode(true);
    //设置线程个数
    ex.set_num_threads(1);
    //将图片放入到网络中,进行前向推理
    ex.input("images", in);
    ncnn::Mat feat_plate;
    ncnn::Mat feat_color;
    //    color = ['黑色', '蓝色', '绿色', '白色', '黄色']
    //获取网络的输出结果
    ex.extract("129", feat_plate);
    ex.extract("output_2", feat_color);
    
    //    int color_index =  max_element(feat_color + 0, feat_color + 78) - feat_color;
//    cout <<feat_color.c << endl;
//    cout <<feat_color.h << endl;
//    cout <<feat_color.w << endl;
    //    pretty_print(feat_color);
    
    ncnn::Mat plate_mat = feat_plate;
    ncnn::Mat color_mat = feat_color;
    vector<string> final_plate_str{};
    
    string finale_plate;
    for (int q = 0; q < plate_mat.c; q++)
    {
        float prebs[21];
        for (int x = 0; x < plate_mat.w; x++)  //遍历21个车牌位置
        {
            const float* ptr = plate_mat.channel(q);
            float preb[78];
            for (int y = 0; y < plate_mat.h; y++)  //遍历78个字符串位置
            {
                preb[y] = ptr[x];  //将18个
                ptr += plate_mat.w;
            }
            int max_num_index = max_element(preb + 0, preb + 78) - preb;
            //            cout<<"max_num_index"<<max_num_index<<endl;
            prebs[x] = max_num_index;
        }
        
        //去重复、去空白a
        vector<int> no_repeat_blank_label{};
        int pre_c = prebs[0];
        cout<<"pre_c"<<pre_c<<endl;
        if (pre_c != 0) {
            no_repeat_blank_label.push_back(pre_c);
        }
        for (int value : prebs)
        {
            if (value == 0 or value==pre_c) {
                if (value == 0 or value == pre_c) {
                    pre_c = value;
                }
                continue;
            }
            no_repeat_blank_label.push_back(value);
            pre_c = value;
        }
        
        // 下面进行车牌lable按照字典进行转化为字符串
        string no_repeat_blank_c = "";
        for (int hh : no_repeat_blank_label) {
            no_repeat_blank_c += plate_chars[hh];
        }
        cout << "单个车牌:" << no_repeat_blank_c << endl;
        
        final_plate_str.push_back(no_repeat_blank_c);
        for (string plate_char : final_plate_str) {
            cout << "所有车牌:" << plate_char << endl;
            finale_plate += plate_char;
        }
    }
    string str = finale_plate;
    cout << str << endl;
    
    float color_result[5];
    for (int q = 0; q < color_mat.c; q++)
    {
        const float* ptr = color_mat.channel(q);
        for (int y = 0; y < color_mat.h; y++)
        {
            
            for (int x = 0; x < color_mat.w; x++)
            {
                //                printf("%f ", ptr[x]);
                //cout << "1111:" << ptr[x];
                color_result[x] = ptr[x];
            }
            ptr += color_mat.w;
            //            printf("\n");
        }
        printf("------------------------\n");
    }
    int color_code = max_element(color_result, color_result + 5) - color_result;
    string color_names[5] = {
        "黑色", "蓝色", "绿色", "白色", "黄色"
    };
//    string color_names[3] = {
//        "蓝", "绿", "黄"
//    };
    
    vector<string> plate_color(2);
    plate_color[0] = str;
    plate_color[1] = color_names[color_code];
    cout << "车牌颜色:" << plate_color[1] << endl;

    return plate_color;
}

#pragma mark - 可视化
// 绘制检测结果
UIImage * Yolov5::draw(UIImage *image, const std::vector<std::string>& classNames, const std::vector<Object>& objects){
    //to do image 转 rgb
    cv::Mat rgb = UIImageToMats(image);
    //    cv::Mat matImage = drawWithMat(rgb, classNames, objects);
    cv::Mat matImage = drawWithMatAsync(rgb, classNames, objects);
    // 将cv::Mat转换回UIImage
    UIImage* resImage = MatToUIImage(matImage);
    return resImage;
}
cv::Mat Yolov5::drawWithMatAsync(cv::Mat image, const std::vector<std::string>& classNames, const std::vector<Object>& objects, float minProb) {
    int i = 0;
    std::vector<std::future<void>> futures;
    
    for (const Object& obj : objects) {
        if (obj.prob < minProb) {
            continue;
        }
        
        cv::Scalar color;
        color = cv::Scalar(56, 177, 11);
      
        futures.push_back(std::async(std::launch::async, &Yolov5::drawCircles, this, std::ref(image), obj, color));
        i++;
    }
    
    for (auto& future : futures) {
        future.wait(); // Wait for the thread to complete
    }
    
    return image;
}
void Yolov5::drawCircles(cv::Mat& image, const Object& obj, const cv::Scalar& color) {
    std::mutex mtx;
    mtx.lock();
    // Perform drawing calculations
    int centerX = static_cast<int>(obj.x + obj.w / 2);
    int centerY = static_cast<int>(obj.y + obj.h / 2);
    int radius = static_cast<int>(std::max(obj.w, obj.h) / 2);
    
    cv::circle(image, cv::Point(centerX, centerY), radius, color, 4);
    mtx.unlock();
}
