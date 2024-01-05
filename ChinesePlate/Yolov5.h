//
//  Yolov5.hpp
//  ChinesePlate
//
//  Created by Nature on 2023/12/28.
//  Copyright © 2023 lprSample. All rights reserved.
//

#ifndef Yolov5_hpp
#define Yolov5_hpp

#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <ncnn/ncnn/net.h>

using namespace std;

struct Object
{
    float x;
    float y;
    float w;
    float h;
    string label;
    string color;
    float p1x;
    float p1y;
    float p2x;
    float p2y;
    float p3x;
    float p3y;
    float p4x;
    float p4y;
    
    float prob;
};

class Yolov5 {
    
public:
    Yolov5();
    ~Yolov5();
    //    int load(const char* modeltype, bool use_gpu = false);
    int load(int target_size = 640, bool use_gpu = false);
    /**
     * 执行目标检测并返回检测到的目标信息。
     *
     * @param image - 输入图像，通常是一张待检测的图像。
     * @param prob_threshold - 概率阈值，用于筛选高概率的目标检测结果（默认为0.6）。
     * @param nms_threshold - 非最大抑制阈值，用于去除重叠的检测结果（默认为0.7）。
     * @return 返回包含检测结果的对象列表，每个对象包括矩形框、类别标签和得分。
     */
    std::vector<Object>detect(UIImage *image,std::string &plateStr,std::string &plateColor, float prob_threshold = 0.6, float nms_threshold = 0.45f);
    std::vector<Object>detectWithMat(cv::Mat image,std::string &plateStr,std::string &plateColor, float prob_threshold = 0.6f, float nms_threshold = 0.45f);
    
    UIImage * draw(UIImage *image, const std::vector<std::string>& classNames, const std::vector<Object>& objects);
    cv::Mat drawWithMatAsync(cv::Mat image, const std::vector<std::string>& classNames, const std::vector<Object>& objects, float minProb= 0.0);
    void drawCircles(cv::Mat& image, const Object& obj, const cv::Scalar& color);
    std::vector<std::string> crnn_rec(const cv::Mat& bgr);

private:
    
    ncnn::Net yolov5;
    int target_size;
    ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
    ncnn::PoolAllocator g_workspace_pool_allocator;
    ncnn::Net crnn;
    ncnn::Net color_net;

};







#endif /* Yolov5_hpp */
