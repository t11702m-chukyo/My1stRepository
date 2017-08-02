#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <opt_flow/Labeling.h>

static const std::string OPENCV_WINDOW = "Image window";
using namespace cv;
using namespace cv::superres;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  cv::Mat PreImg;
  Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();

public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/usb_cam/image_raw", 1,
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    // 入力画像の取得
    cv_bridge::CvImagePtr InImgRos;
    try
    {
      InImgRos = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Optical Flow ====================================================================
    // グレイスケール変換
    cv_bridge::CvImage GrayImgRos;
    cv::Mat GrayImg;
    cv::cvtColor(InImgRos->image, GrayImgRos.image, CV_BGR2GRAY);

    // オプティカルフロー用の画像（1/10倍のりサイズ+グレイスケール）の作成
    GrayImg = GrayImgRos.image.clone();
    cv::resize( GrayImg, GrayImg, cv::Size(), 0.1, 0.1 );
    int width   = GrayImg.cols;
    int height  = GrayImg.rows;
    // =================================================================================
      
    // Optical Flow ====================================================================
    if(PreImg.empty()) PreImg = GrayImg.clone();
		cv::Mat flowX, flowY;
		cv::Mat magnitude, angle;
		cv::Mat hsvPlanes[3];		
		cv::Mat hsv;
		cv::Mat flowBgr;
    cv::Mat BinImg = cv::Mat::zeros(height, width, CV_8U);
    int WhiteNum;

    // オプティカルフローの算出
		opticalFlow->calc(PreImg, GrayImg, flowX, flowY);

		// オプティカルフローの強度画像(magnitude)の作成
		cartToPolar(flowX, flowY, magnitude, angle, true);
		normalize(magnitude, magnitude, 0, 1, NORM_MINMAX); // 正規化

    // オプティカルフローの強度画像の2値化
    WhiteNum = 0;
    for(int j = 0; j < height; j++){
      for(int i = 0; i < width; i++){
        if(0.5 < magnitude.at<float>(j, i)){
          BinImg.at<unsigned char>(j, i) = 255;
          WhiteNum++;
        }
      }
    }

    // 強度画像の補正（穴埋め）
    cv::dilate(BinImg, BinImg, cv::Mat(), cv::Point(-1, -1), 3); // クロージングを3回実施
    cv::erode(BinImg, BinImg, cv::Mat(), cv::Point(-1, -1), 3);  // オープニングを3回実施

    // =================================================================================

    // 物体領域の表示 ========================================================================
    short         *LabelImg = new short[width*height];  // ラベリング結果
    int           min_size = 20;                        // 最小セグメントのサイズ
    bool          labeling_flag;                        // ラベリング実行の有無を示すフラグ（）
    cv::Mat ResImg = InImgRos->image.clone();           // 物体検出結果 
    cv::Point     bb_pos1;                              // Bounding Boxの左上の位置
    cv::Point     bb_pos2;                              // Bounding Boxの右下の位置
    LabelingBS    labeling;                             // ラベリング結果
    RegionInfoBS  *ri;

    if(min_size < WhiteNum) labeling_flag = true;
    else                    labeling_flag = false;

    if(labeling_flag){
      // Labeling の実行
      labeling.Exec((unsigned char *)BinImg.data, LabelImg, width, height, true, min_size);

      // 最大サイズのセグメントのBounding Boxの位置(bb_pos)を取得
      ri = labeling.GetResultRegionInfo(0); // 最大サイズのセグメントのみの情報を取得
      ri->GetMin(bb_pos1.x, bb_pos1.y);
      ri->GetMax(bb_pos2.x, bb_pos2.y);

      // 物体検出結果の表示
      bb_pos1.x = bb_pos1.x*10; bb_pos1.y = bb_pos1.y*10;;
      bb_pos2.x = bb_pos2.x*10; bb_pos2.y = bb_pos2.y*10;;
      cv::rectangle( ResImg, bb_pos1, bb_pos2, cv::Scalar(0, 255, 255), 1, 8, 0 );
    }
    // =================================================================================

		// 前のフレームを保存
		PreImg = GrayImg;

		// 表示
    cv::imshow("result", ResImg);
    cv::waitKey(1);
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
