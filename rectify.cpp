#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"

#include "rectify.h"

Rectify::Rectify() {
    stereoRectify(K1, D1, K2, D2, image_size, R, T,
                  R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0.5, image_size, &roi1, &roi2);
};

Rectify::~Rectify() {};

std::shared_ptr<cv::Mat> Rectify::AlignImage(const cv::Mat &image) {
    cv::Mat output = image.clone();
    cv::Mat left_map_x, left_map_y, right_map_x, right_map_y;
    cv::Rect left_roi(0, 0, image_size.width, image_size.height);
    cv::Rect right_roi(image_size.width, 0, image_size.width, image_size.height);

    cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32F, left_map_x, left_map_y);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_32F, right_map_x, right_map_y);
    cv::remap(image(left_roi), output(left_roi), left_map_x, left_map_y, cv::INTER_LINEAR);
    cv::remap(image(right_roi), output(right_roi), right_map_x, right_map_y, cv::INTER_LINEAR);
    return std::make_shared<cv::Mat>(output);
}

void Rectify::DrawLine(cv::Mat &image) {
    for (int i = 0; i < image.rows; i += 20) {
        cv::line(image, cv::Point(0, i), cv::Point(image.cols, i), cv::Scalar(0, 255, 0));
    }
}

std::shared_ptr<cv::Mat> Rectify::DrawGoodMatches(const cv::Mat &img1,
                                                  const cv::Mat &img2,
                                                  const std::vector<cv::KeyPoint> &keypoints1,
                                                  const std::vector<cv::KeyPoint> &keypoints2,
                                                  std::vector<cv::DMatch> &matches) {
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;

    const int ptsPairs = std::min(10, (int) (matches.size() * 0.15));
    for (int i = 0; i < ptsPairs; i++) {
        good_matches.push_back(matches[i]);
    }
    //std::cout << "Max distance: " << matches.front().distance << std::endl;
    //std::cout << "Min distance: " << matches.back().distance << std::endl;

    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2,
                    good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return std::make_shared<cv::Mat>(img_matches);
}

std::shared_ptr<cv::Mat> Rectify::GetAlignImage(const cv::Mat &image) {
    auto output = AlignImage(image);
    DrawLine(*output);
    return output;
}

std::shared_ptr<cv::Mat> Rectify::GetMatchedImage(const cv::Mat &image) {
    auto output = AlignImage(image);
    cv::Mat gray_image;
    cv::cvtColor(*output, gray_image, cv::COLOR_BGR2GRAY);
    cv::Range r1(0, output->cols / 2);
    cv::Range r2(output->cols / 2, output->cols);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    auto surf = cv::xfeatures2d::SURF::create(800.0);
    surf->detectAndCompute(gray_image.colRange(r1), cv::Mat(), keypoints1, descriptors1);
    surf->detectAndCompute(gray_image.colRange(r2), cv::Mat(), keypoints2, descriptors2);

    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    output = DrawGoodMatches(output->colRange(r1), output->colRange(r2), keypoints1, keypoints2, matches);
    return output;
}

std::shared_ptr<cv::Mat> Rectify::GetDisparityImage(const cv::Mat &image) {
    const int num_disparities = 16;
    const int block_size = 9;
    auto output = AlignImage(image);
    cv::cvtColor(*output, *output, cv::COLOR_BGR2GRAY);

    auto bm = cv::StereoBM::create(num_disparities, block_size);
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    cv::Mat disp, disp8, disp_color;
    auto t1 = cv::getTickCount();
    bm->compute(output->colRange(0, output->cols / 2), output->colRange(output->cols / 2, output->cols), disp);
    //std::cout << "compute time:" << (cv::getTickCount() - t1) * 1000 / cv::getTickFrequency() << std::endl;

    disp.convertTo(disp8, CV_8U, 255 / (num_disparities * 16.));
    cv::applyColorMap(disp8, disp_color, cv::COLORMAP_JET);
    return std::make_shared<cv::Mat>(disp_color);
}


