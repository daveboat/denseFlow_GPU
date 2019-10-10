#ifndef TOOLBOX_HH
#define TOOLBOX_HH

#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace toolbox{

  inline std::string int_to_string( int64_t val ){
    std::stringstream out;
    out << val;
    return out.str();
  }

  void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y, double lowerBound, double higherBound);

  class IOManager{
    public:
      IOManager(const std::string outFolder);

      void WriteImg( const cv::Mat & img, const int64_t id );
      void WriteFlow( const cv::Mat & flow_x, const cv::Mat & flow_y, const int64_t id);

    private:
      std::string CreateFilename( const int64_t id, const int64_t type );

      std::string outFolder_;
  };

} // namespace toolbox
#endif