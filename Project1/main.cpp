#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>

// Настройки (можно менять)
#define USE_ORIGINAL_POINTS false  // true - использовать точки до эквализации, false - после
#define DRAW_TRACES true           // true - рисовать траектории, false - рисовать точки
#define MAX_TRACE_LENGTH 20        // Максимальная длина траектории (в кадрах)
#define MIN_POINTS_TO_TRACK 1     // Минимальное количество точек для продолжения отслеживания

// Настройки области интереса
#define USE_FIXED_ROI false        // true - использовать фиксированную ROI, false - автоматическое определение
#define FIXED_ROI_X 500            // Левый край ROI
#define FIXED_ROI_Y 1000          // Верхний край ROI
#define FIXED_ROI_WIDTH 1400        // Ширина ROI
#define FIXED_ROI_HEIGHT 600      // Высота ROI

int main(int argc, char** argv)
{
    cv::namedWindow("Processed Video", cv::WINDOW_AUTOSIZE);

    // Настройка стартового кадра
    const int START_FRAME = 0; // Начальный кадр (0 для обработки с начала)
    bool skip_initial_frames = START_FRAME > 320;

    cv::VideoCapture cap("C:/Users/bugro/Videos/VideoTest22.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть видео!" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Size frameSize(frame_width, frame_height);

    cv::VideoWriter writerROI("C:/Users/bugro/Videos/output_roi.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize, true);

    cv::VideoWriter writerTraces("C:/Users/bugro/Videos/output_traces.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize, true);

    std::ofstream outFile("C:/Users/bugro/Videos/features_count.txt");

    cv::Mat frame, prev_frame, prev_gray, gray, edges;
    int frameNum = 0;

    // Для хранения точек и их траекторий
    std::vector<cv::Point2f> prev_corners;
    std::map<int, std::vector<cv::Point2f>> traces;
    int next_point_id = 0;

    // Область интереса
    cv::Rect roi;
    if (USE_FIXED_ROI) {
        // Используем жестко заданную ROI
        roi = cv::Rect(FIXED_ROI_X, FIXED_ROI_Y, FIXED_ROI_WIDTH, FIXED_ROI_HEIGHT);
        std::cout << "Используется фиксированная ROI: x=" << FIXED_ROI_X << " y=" << FIXED_ROI_Y
            << " width=" << FIXED_ROI_WIDTH << " height=" << FIXED_ROI_HEIGHT << std::endl;
    }

    // Функция для инициализации точек отслеживания
    auto initialize_tracking_points = [&](cv::Mat& roi_gray, cv::Mat& eq_roi_gray) {
        std::vector<cv::Point2f> new_corners;
        if (USE_ORIGINAL_POINTS) {
            cv::goodFeaturesToTrack(roi_gray, new_corners, 5000, 0.01, 10);
        }
        else {
            cv::goodFeaturesToTrack(eq_roi_gray, new_corners, 5000, 0.01, 10);
        }

        if (!new_corners.empty()) {
            prev_corners = new_corners;
            traces.clear();
            next_point_id = 0;

            for (const auto& corner : new_corners) {
                traces[next_point_id++] = { corner };
            }

            prev_gray = USE_ORIGINAL_POINTS ? roi_gray.clone() : eq_roi_gray.clone();
            std::cout << "Инициализировано новых точек: " << new_corners.size() << std::endl;
        }
        return !new_corners.empty();
        };

    // Пропускаем кадры до START_FRAME
    if (skip_initial_frames) {
        std::cout << "Пропускаем кадры до " << START_FRAME << std::endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, START_FRAME);
        frameNum = START_FRAME;
    }

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameNum++;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);

        if (!USE_FIXED_ROI) {
            // Автоматическое определение ROI
            cv::Canny(gray, edges, 55, 110);
            cv::dilate(edges, edges, cv::Mat());

            // Определяем верхнюю границу (horizonY)
            int horizonY = 0;
            for (int y = 0; y < frame.rows / 2; y++) {
                if (cv::countNonZero(edges.row(y)) > frame.cols * 0.1) {
                    horizonY = y;
                    break;
                }
            }

            // Определяем нижнюю границу (hoodY)
            int hoodY = horizonY + FIXED_ROI_HEIGHT;
            if (hoodY >= frame.rows) {
                hoodY = frame.rows - 1;
            }

            roi = cv::Rect(0, horizonY, frame.cols, hoodY - horizonY);
        }

        // Проверяем, чтобы ROI не выходила за границы кадра
        if (roi.x < 0) roi.x = 0;
        if (roi.y < 0) roi.y = 0;
        if (roi.x + roi.width > frame.cols) roi.width = frame.cols - roi.x;
        if (roi.y + roi.height > frame.rows) roi.height = frame.rows - roi.y;

        cv::Mat roi_gray = gray(roi).clone();
        cv::Mat eq_roi_gray;
        cv::equalizeHist(roi_gray, eq_roi_gray);

        // Если это первый кадр обработки или точек для отслеживания слишком мало
        if ((skip_initial_frames && frameNum == START_FRAME + 1) ||
            (!skip_initial_frames && frameNum == 1) ||
            (prev_corners.size() < MIN_POINTS_TO_TRACK)) {

            if (!initialize_tracking_points(roi_gray, eq_roi_gray)) {
                std::cerr << "Не удалось инициализировать точки для отслеживания на кадре " << frameNum << std::endl;
                if (skip_initial_frames && frameNum == START_FRAME + 1) {
                    skip_initial_frames = false;
                    continue;
                }
            }

            if (skip_initial_frames && frameNum == START_FRAME + 1) {
                skip_initial_frames = false;
            }

            prev_frame = frame.clone();
            continue;
        }

        // Подготавливаем текущее изображение для отслеживания
        cv::Mat current_gray = USE_ORIGINAL_POINTS ? roi_gray : eq_roi_gray;

        // На последующих кадрах отслеживаем точки
        std::vector<cv::Point2f> curr_corners;
        std::vector<uchar> status;
        std::vector<float> err;

        try {
            cv::calcOpticalFlowPyrLK(
                prev_gray,
                current_gray,
                prev_corners,
                curr_corners,
                status,
                err,
                cv::Size(21, 21),
                5,
                cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.3)
            );
        }
        catch (const cv::Exception& e) {
            std::cerr << "Ошибка в calcOpticalFlowPyrLK на кадре " << frameNum << ": " << e.what() << std::endl;
            initialize_tracking_points(roi_gray, eq_roi_gray);
            continue;
        }

        // Обновляем траектории
        std::vector<cv::Point2f> good_new;
        std::map<int, std::vector<cv::Point2f>> new_traces;
        int point_index = 0;
        int kept_points = 0;

        for (auto& trace : traces) {
            if (point_index >= status.size()) break;

            if (status[point_index]) {
                cv::Point2f new_point = curr_corners[point_index];
                if (new_point.x >= 0 && new_point.x < roi.width &&
                    new_point.y >= 0 && new_point.y < roi.height) {

                    trace.second.push_back(new_point);

                    if (trace.second.size() > MAX_TRACE_LENGTH) {
                        trace.second.erase(trace.second.begin());
                    }

                    new_traces[trace.first] = trace.second;
                    good_new.push_back(new_point);
                    kept_points++;
                }
            }
            point_index++;
        }

        traces = new_traces;
        prev_corners = good_new;
        prev_gray = current_gray.clone();
        prev_frame = frame.clone();

        // Запись количества точек в файл
        outFile << frameNum << " " << kept_points << std::endl;

        // Рисуем результаты
        cv::Mat result_frame = frame.clone();

        if (DRAW_TRACES) {
            for (const auto& trace : traces) {
                const auto& points = trace.second;
                if (points.size() < 2) continue;

                for (size_t i = 1; i < points.size(); i++) {
                    cv::line(
                        result_frame,
                        cv::Point(points[i - 1].x + roi.x, points[i - 1].y + roi.y),
                        cv::Point(points[i].x + roi.x, points[i].y + roi.y),
                        cv::Scalar(0, 255, 0),
                        2
                    );
                }

                cv::circle(
                    result_frame,
                    cv::Point(points.back().x + roi.x, points.back().y + roi.y),
                    4,
                    cv::Scalar(0, 0, 255),
                    -1
                );
            }
        }
        else {
            for (const auto& corner : prev_corners) {
                cv::circle(
                    result_frame,
                    cv::Point(corner.x + roi.x, corner.y + roi.y),
                    4,
                    cv::Scalar(0, 255, 0),
                    -1
                );
            }
        }

        // Рисуем область интереса
        cv::rectangle(
            result_frame,
            roi,
            cv::Scalar(255, 0, 0),
            2
        );

        writerROI.write(result_frame);
        writerTraces.write(result_frame);
        cv::imshow("Processed Video", result_frame);

        if (cv::waitKey(33) >= 0) break;
    }

    cap.release();
    writerROI.release();
    writerTraces.release();
    outFile.close();
    cv::destroyAllWindows();

    return 0;
}