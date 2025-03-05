#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main(int argc, char** argv)
{
	cv::namedWindow("Example3", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Procesed Video", cv::WINDOW_AUTOSIZE);

	cv::VideoCapture cap;
	cap.open("C:/Users/bugro/Videos/VideoTest22.mp4");

	double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    cv::VideoWriter writer("C:/Users/bugro/Videos/output.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize, false);

    cv::Mat frame, edges;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, edges, cv::COLOR_BGR2GRAY);
        cv::Canny(edges, edges, 100, 200);

        cv::imshow("Original Video", frame);
        cv::imshow("Processed Video", edges);

        writer.write(edges);

        if (cv::waitKey(33) >= 0) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}