#include "toolbox.hh"
//#include <iostream>

namespace toolbox {
	void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y, double lowerBound, double higherBound) {
		//convertTo is output(x,y) = alpha * image(x,y) + beta
		double alpha = 256.0/(higherBound - lowerBound);
		double beta = 128;
		flow_x.convertTo( img_x, CV_8UC1, alpha, beta );
		flow_y.convertTo( img_y, CV_8UC1, alpha, beta );
	}

	IOManager::IOManager( const std::string outFolder){
		outFolder_ = outFolder;
	}

	void IOManager::WriteImg( const cv::Mat & img, const int64_t id ){
		cv::imwrite( CreateFilename( id, 0 ), img );
	}

	void IOManager::WriteFlow( const cv::Mat & flow_x, const cv::Mat & flow_y, const int64_t id){
		cv::imwrite( CreateFilename( id, 1 ), flow_x );
		cv::imwrite( CreateFilename( id, 2 ), flow_y );
	}

	std::string IOManager::CreateFilename( const int64_t id, const int64_t type ){
		std::string name = "";

		switch( type ){
			case 0:
				name = outFolder_ + "/rgb/" + int_to_string( id ) + ".jpg";
				break;
			case 1:
				name = outFolder_ + "/flow_x/" + int_to_string( id ) + ".jpg";
				break;
			case 2:
				name = outFolder_ + "/flow_y/" + int_to_string( id ) + ".jpg";
				break;
			default:
				name = "";
		}
    
		return name;
	}

}	// namespace toolbox