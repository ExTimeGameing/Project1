#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"

int main(int argc, char** argv)
{
    cv::namedWindow("Processed Video", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap;
    cap.open("C:/Users/bugro/Videos/VideoTest22.mp4");

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    cv::VideoWriter writerROI("C:/Users/bugro/Videos/output_roi.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize, true);

    cv::Mat frame, edges, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5); // Другой способ убрать шумы кроме размытия ?
        cv::Canny(gray, edges, 55, 110);
        cv::dilate(edges, edges, cv::Mat());  // Утолщаем линии для лучшей видимости

        writerROI.write(edges);

        int horizonY = 0;
        for (int y = 0; y < frame.rows / 2; y++) {
            if (cv::countNonZero(edges.row(y)) > frame.cols * 0.1) {
                horizonY = y;
                break;
            }
        }

        int hoodY = 3 * frame.rows / 4;  // Стартуем поиск с 3/4 кадра
        for (int y = 3 * frame.rows / 4; y < frame.rows - 1; y++) {
            if (cv::countNonZero(edges.row(y)) > frame.cols * 0.05) {
                hoodY = y;
                break;
            }
        }

        // Рисуем область интереса
        cv::rectangle(frame, cv::Point(0, horizonY), cv::Point(frame.cols, hoodY), cv::Scalar(0, 255, 0), 2);

        // Для дебага, незабыть закомментить накладываем контуры Кэнни поверх кадра
        cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
        cv::addWeighted(frame, 0.8, edges, 0.5, 0, frame);

        writerROI.write(frame);
        cv::imshow("Processed Video", frame);

        if (cv::waitKey(33) >= 0) break;
    }

    cap.release();
    writerROI.release();
    cv::destroyAllWindows();

    return 0;
}