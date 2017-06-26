#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/opencv.hpp>

static const std::string OPENCV_WINDOW = "Image window";
using namespace cv;
using namespace cv::superres;

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  Mat prev;
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

// add
	void visualizeFarnebackFlow(
    const Mat& flow,    //オプティカルフロー CV_32FC2
    Mat& visual_flow    //可視化された画像 CV_32FC3
	)
	{
    visual_flow = Mat::zeros(flow.rows, flow.cols, CV_32FC3);
    int flow_ch = flow.channels();
    int vis_ch = visual_flow.channels();//3のはず
    for(int y = 0; y < flow.rows; y++) {
        float* psrc = (float*)(flow.data + flow.step * y);
        float* pdst = (float*)(visual_flow.data + visual_flow.step * y);
        for(int x = 0; x < flow.cols; x++) {
            float dx = psrc[0];
            float dy = psrc[1];
            float r = (dx < 0.0) ? abs(dx) : 0;
            float g = (dx > 0.0) ? dx : 0;
            float b = (dy < 0.0) ? abs(dy) : 0;
            r += (dy > 0.0) ? dy : 0;
            g += (dy > 0.0) ? dy : 0;
 
            pdst[0] = b;
            pdst[1] = g;
            pdst[2] = r;
 
            psrc += flow_ch;
            pdst += vis_ch;
        }
    }
	}


  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    // ######### 0529 add -start #########
    cv_bridge::CvImage gray;
    cv::Mat curr;
    cv::cvtColor(cv_ptr->image, gray.image, CV_BGR2GRAY);

    curr = gray.image.clone();
    cv::resize( curr, curr, cv::Size(), 0.5, 0.5 );
      
    if(prev.empty()) prev = curr.clone();
    // オプティカルフローの計算
		Mat flowX, flowY;
		opticalFlow->calc(prev, curr, flowX, flowY);

		// オプティカルフローの可視化（色符号化）
		//  オプティカルフローを極座標に変換（角度は[deg]）
		Mat magnitude, angle;
		cartToPolar(flowX, flowY, magnitude, angle, true);
		//  色相（H）はオプティカルフローの角度
		//  彩度（S）は0～1に正規化したオプティカルフローの大きさ
		//  明度（V）は1
		Mat hsvPlanes[3];		
		hsvPlanes[0] = angle;
		normalize(magnitude, magnitude, 0, 1, NORM_MINMAX); // 正規化
		hsvPlanes[1] = magnitude;
		hsvPlanes[2] = Mat::ones(magnitude.size(), CV_32F);
		//  HSVを合成して一枚の画像にする
		Mat hsv;
		merge(hsvPlanes, 3, hsv);
		//  HSVからBGRに変換
		Mat flowBgr;
		cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

		// 表示
		cv::imshow("input", curr);
		cv::imshow("optical flow", flowBgr);

		// 前のフレームを保存
		prev = curr;

    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, gray.image);
    cv::waitKey(3);

    // Output modified video stream
    //image_pub_.publish(gray2.toImageMsg()); \\ COLOR
    image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", gray.image).toImageMsg()); // GRAYSCALE

  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
