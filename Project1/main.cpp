#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
    cv::namedWindow("Processed Video", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap("C:/Users/bugro/Videos/VideoTest22.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть видео!" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    cv::VideoWriter writerROI("C:/Users/bugro/Videos/output_roi.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize, true);

    std::ofstream outFile("C:/Users/bugro/Videos/features_count.txt");

    cv::Mat frame, edges, gray, roi_gray, eq_roi_gray;
    int frameNum = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameNum++;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);
        cv::Canny(gray, edges, 55, 110);
        cv::dilate(edges, edges, cv::Mat());

        // Найти границы области интереса
        int horizonY = 0;
        for (int y = 0; y < frame.rows / 2; y++) {
            if (cv::countNonZero(edges.row(y)) > frame.cols * 0.1) {
                horizonY = y;
                break;
            }
        }

        int hoodY = 3 * frame.rows / 4;
        for (int y = 3 * frame.rows / 4; y < frame.rows - 1; y++) {
            if (cv::countNonZero(edges.row(y)) > frame.cols * 0.05) {
                hoodY = y;
                break;
            }
        }

        // Вырезаем область интереса
        cv::Rect roi(0, horizonY, frame.cols, hoodY - horizonY);
        roi_gray = gray(roi);

        // Задание 1 - Найти угловые признаки без эквализации
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(roi_gray, corners, 5000, 0.01, 10);

        // Эквализация гистограммы
        cv::equalizeHist(roi_gray, eq_roi_gray);

        // Задание 2 - Найти угловые признаки после эквализации
        std::vector<cv::Point2f> eq_corners;
        cv::goodFeaturesToTrack(eq_roi_gray, eq_corners, 5000, 0.01, 10);

        // Запись количества точек в файл
        outFile << frameNum << " " << corners.size() << " " << eq_corners.size() << std::endl;

        // Рисуем найденные точки на кадре
        for (auto& corner : corners) {
            cv::circle(frame, cv::Point(corner.x, corner.y + horizonY), 4, cv::Scalar(0, 255, 0), -1);
        }
        for (auto& corner : eq_corners) {
            cv::circle(frame, cv::Point(corner.x, corner.y + horizonY), 4, cv::Scalar(0, 0, 255), -1);
        }

        // Рисуем область интереса
        cv::rectangle(frame, cv::Point(0, horizonY), cv::Point(frame.cols, hoodY), cv::Scalar(255, 0, 0), 2);

        writerROI.write(frame);
        cv::imshow("Processed Video", frame);

        if (cv::waitKey(33) >= 0) break;
    }

    cap.release();
    writerROI.release();
    outFile.close();
    cv::destroyAllWindows();

    return 0;
}