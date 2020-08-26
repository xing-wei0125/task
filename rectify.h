#include <memory>
#include <vector>
#include <opencv2/core.hpp>

class Rectify {
public:
    Rectify();

    ~Rectify();

    /**
     * Rectify stereo image and draw line
     * @param src_image
     * @return dst_image
     */
    std::shared_ptr<cv::Mat> GetAlignImage(const cv::Mat &image);

    /**
     * Detect and match features of stereo image and draw matched points
     * @param src_image
     * @return dst_image
     */
    std::shared_ptr<cv::Mat> GetMatchedImage(const cv::Mat &image);

    /**
     * Get the disparity image of stereo image
     * @param src_image
     * @return dst_image
     */
    std::shared_ptr<cv::Mat> GetDisparityImage(const cv::Mat &image);

private:
    /**
     * Rectify stereo image
     * @param src_image
     * @return dst_image
     */
    std::shared_ptr<cv::Mat> AlignImage(const cv::Mat &image);

    /**
     * Draw lines
     * @param src_image
     */
    void DrawLine(cv::Mat &image);

    /**
     * Draw matched points
     * @param img1, the left image of the stereo image
     * @param img2, the right image of the stereo image
     * @param keypoints1, the keypoints detected from the left image
     * @param keypoints2, the keypoints detected from the right image
     * @param matches, matched points
     * @return dst_image
     */
    std::shared_ptr<cv::Mat> DrawGoodMatches(const cv::Mat &img1,
                                             const cv::Mat &img2,
                                             const std::vector<cv::KeyPoint> &keypoints1,
                                             const std::vector<cv::KeyPoint> &keypoints2,
                                             std::vector<cv::DMatch> &matches);

    const cv::Mat K1 = (cv::Mat_<double>(3, 3) <<
                                               530.90002, 0, 136.63037, 0, 581.00362, 161.32884, 0, 0, 1);
    const cv::Mat D1 = (cv::Mat_<double>(1, 5) <<
                                               -0.28650, 0.29524, -0.00212, 0.00152, 0.0);
    const cv::Mat K2 = (cv::Mat_<double>(3, 3) <<
                                               524.84413, 0, 216.17358, 0, 577.11024, 149.76379, 0, 0, 1);
    const cv::Mat D2 = (cv::Mat_<double>(1, 5) <<
                                               -0.25745, 0.62307, 0.03660, -0.01082, 0.0);
    const cv::Mat R = (cv::Mat_<double>(3, 3) <<
                                              0.9990, -0.0112, -0.0426, 0.0117, 0.9999, 0.0097, 0.0425, -0.0102, 0.9990);
    const cv::Mat T = (cv::Mat_<double>(3, 1) <<
                                              -5.49238, 0.04267, -0.39886);
    const cv::Size image_size = cv::Size(360, 288);

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
    cv::Rect roi1;
    cv::Rect roi2;
};
