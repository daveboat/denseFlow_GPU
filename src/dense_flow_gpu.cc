#include "dense_flow_gpu.hh"
#include <sys/stat.h>

// Compute optical flow between frames t and t+steppings[*]
const std::vector< int64_t > flow_span { 1 };

cv::Ptr<cv::cuda::FarnebackOpticalFlow> alg_farn;
cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> alg_tvl1;
cv::Ptr<cv::cuda::BroxOpticalFlow> alg_brox;

int main(int argc, char** argv){  
  // Get args
  const cv::String keys =
  "{ f vidFile     | | Video filename}"
  "{ o outFolder   | | output folder - rbg saved in outFolder/rgb/, flow saved in outFolder/flow_x and outFolder/flow_y}"
  "{ r resize      | 256 | resize video so that smaller of width and height is this number of pixels, with bilinear interpolation and preserving aspect ratio. set <=0 for no resize (Default = 256)}"
  "{ b bound       | 20 | optical flow value upper and lower limit: values outside of (-bound, bound) are truncated. (Default = 20)}"
  "{ t type        | 1 | optical flow algorithm (0 = Farneback, 1 = TVL1, 2 = Brox). (Default = 1)}"
  "{ d device_id   | 0 | gpu id to use (Default = 0)}"
  "{ s step        | 1 | number of frames to skip when saving optical flow and rgb frames.(Default = 1)}";

  cv::CommandLineParser cmd(argc, argv, keys);

  std::string vidFile     = cmd.get<std::string>("vidFile");
  std::string outFolder   = cmd.get<std::string>("outFolder");
  int resize              = cmd.get<int>("resize");
  int bound               = cmd.get<int>("bound");
  int type                = cmd.get<int>("type");
  int device_id           = cmd.get<int>("device_id");
  int step                = cmd.get<int>("step");

  if( !cmd.check() ){
    cmd.printErrors();
    return -1;
  }

  //try to create output directories
  CreateOutputDirectories(outFolder);

  //set gpu
  cv::cuda::setDevice( device_id );

  //initialize flow algorithm
  switch( type ){
    case 0:
      alg_farn = cv::cuda::FarnebackOpticalFlow::create();
      break;
    case 1:
      alg_tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
      break;
    case 2:
      alg_brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
      break;
    default:
      alg_brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
  }

  //open video
  Video v( vidFile, step, resize);    // Sample every `step`-th frame, up to the `len_clip`-th frame, i.e., floor(len_clip/step) frames total.

  if( !v.is_open() )
    //std::cout << "Could not open video " << vidFile << "\n";
    return -1;

  //begin processing video
  toolbox::IOManager io_manager( outFolder );
  ProcessClip( v, io_manager, type, bound, resize );

  return 0;
}

void CreateOutputDirectories(std::string outFolder){
  std::string outFolderRGB = outFolder + "/rgb";
  std::string outFolderFlow_x = outFolder + "/flow_x";
  std::string outFolderFlow_y = outFolder + "/flow_y";
  mkdir(outFolder.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
  mkdir(outFolderRGB.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
  mkdir(outFolderFlow_x.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
  mkdir(outFolderFlow_y.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
}

void ProcessClip( Video & v, toolbox::IOManager & io_manager, const int type, const int bound, const int resize ){
  auto max_span = std::max_element( flow_span.begin(), flow_span.end() );    

  std::vector< std::pair<int64_t, cv::Mat> > clip;

  //read first image for flow calculation
  v.read( clip, *max_span, true );

  int64_t counter = 0;

  //loop through image
  while( true ){
    v.read( clip, 1, true );

    if( clip.empty() ){
      break;
    }

    counter++;

    //write RGB image
    io_manager.WriteImg( clip[0].second, counter );
    
    for( int i = 0; i < flow_span.size(); i++ ){
      int span = flow_span[i];

      if( span >= clip.size() )
        continue;

      //opencv can't save images with 2 channels, rofl
      cv::Mat flow_x, flow_y;

      cv::Mat grey_first, grey_second;
      cv::cvtColor( clip.begin()->second,         grey_first, cv::COLOR_BGR2GRAY );
      cv::cvtColor( (clip.begin()+span)->second,  grey_second, cv::COLOR_BGR2GRAY );

      //compute flow and convert to 8 bit images
      ComputeFlow( grey_first, grey_second, type, bound, flow_x, flow_y);

      //write flow images
      io_manager.WriteFlow(flow_x, flow_y, counter);
    }

    clip.erase(clip.begin(), clip.begin()+1 );
  }
}

void ComputeFlow( const cv::Mat prev, const cv::Mat cur, const int type, const int bound, cv::Mat & flow_x, cv::Mat & flow_y ){

  // GPU optical flow
  cv::cuda::GpuMat frame_0( prev );
  cv::cuda::GpuMat frame_1( cur );

  cv::cuda::GpuMat d_flow( frame_0.size(), CV_32FC2 );

  switch(type){
    case 0:
      alg_farn->calc(frame_0, frame_1, d_flow );
      break;
    case 1:
      alg_tvl1->calc(frame_0, frame_1, d_flow );
      break;
    case 2:
      cv::cuda::GpuMat d_frame0f, d_frame1f;
      frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
      frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

      alg_brox->calc(d_frame0f, d_frame1f, d_flow );
      break;
  }

  //split channels, apply boundary, and convert to 8 bit single-channel images (0 to 255)
  cv::Mat flow( d_flow );
  cv::Mat flow_channels[2];

  cv::split( flow, flow_channels);

	toolbox::convertFlowToImage( flow_channels[0], flow_channels[1], flow_x, flow_y, -bound, bound );
}