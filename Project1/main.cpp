#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

// Функция для создания цветного представления оптического потока (Hue-Saturation-Value)
Mat drawOpticalFlowColor(const Mat& flow) {
    Mat flow_rgb(flow.size(), CV_8UC3);
    Mat hsv(flow.size(), CV_8UC3);

    // Создаем изображение HSV: H - направление, S - 255 (максимальная насыщенность), V - длина вектора
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            float fx = flow.at<Vec2f>(y, x)[0];
            float fy = flow.at<Vec2f>(y, x)[1];

            // Вычисляем угол и длину вектора
            double angle = atan2(fy, fx); // Угол в радианах
            double magnitude = sqrt(fx * fx + fy * fy);

            // Нормализуем угол в диапазон [0, 2*PI] -> [0, 180] для Hue
            int hue = static_cast<int>(((angle + CV_PI) / (2 * CV_PI)) * 180.0);
            // Нормализуем длину в диапазон [0, 255] для Value
            int value = min(static_cast<int>(magnitude * 10), 255); // Умножаем на коэффициент для видимости

            // Заполняем HSV
            hsv.at<Vec3b>(y, x) = Vec3b(hue, 255, value);
        }
    }

    // Конвертируем HSV в BGR для отображения
    cvtColor(hsv, flow_rgb, COLOR_HSV2BGR);
    return flow_rgb;
}

// Функция для создания векторного представления оптического потока (отрезки с окружностями)
Mat drawOpticalFlowVectors(const Mat& flow, int step = 20) {
    Mat flow_vectors(flow.size(), CV_8UC3);
    flow_vectors.setTo(Scalar(0, 0, 0)); // Черный фон

    // Находим максимальную длину вектора для масштабирования
    float maxMagnitude = 0.0f;
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            float fx = flow.at<Vec2f>(y, x)[0];
            float fy = flow.at<Vec2f>(y, x)[1];
            float magnitude = sqrt(fx * fx + fy * fy);
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
            }
        }
    }

    // Масштабный коэффициент: самый длинный вектор должен быть равен step пикселам
    float scale = (maxMagnitude > 0) ? static_cast<float>(step) / maxMagnitude : 1.0f;

    // Рисуем векторы на сетке с шагом `step`
    for (int y = step / 2; y < flow.rows; y += step) {
        for (int x = step / 2; x < flow.cols; x += step) {
            float fx = flow.at<Vec2f>(y, x)[0];
            float fy = flow.at<Vec2f>(y, x)[1];

            // Масштабируем вектор
            float scaled_fx = fx * scale;
            float scaled_fy = fy * scale;

            // Координаты конца вектора
            Point end_point(x + static_cast<int>(scaled_fx), y + static_cast<int>(scaled_fy));

            // Рисуем белый отрезок
            line(flow_vectors, Point(x, y), end_point, Scalar(255, 255, 255), 1, LINE_AA);

            // Рисуем белую окружность в конце вектора
            circle(flow_vectors, end_point, 2, Scalar(255, 255, 255), -1, LINE_AA);
        }
    }

    return flow_vectors;
}

// Функция для обработки видеофайла с заданными параметрами
void processVideo(const string& inputPath, const string& outputPathColor, const string& outputPathVectors, const string& outputPathDetection, const string& timeLogPath, double pyrScale, int levels, int winsize, int iterations, int polyN, double polySigma, int flags) {
    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл: " << inputPath << endl;
        return;
    }

    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    Size frameSize(static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)));

    // Видео с цветным представлением оптического потока
    VideoWriter writerColor(outputPathColor, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    // Видео с векторным представлением оптического потока
    VideoWriter writerVectors(outputPathVectors, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    // Видео с выделением движущихся объектов в области интереса
    VideoWriter writerDetection(outputPathDetection, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);

    if (!writerColor.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для цветного потока: " << outputPathColor << endl;
        return;
    }
    if (!writerVectors.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для векторного потока: " << outputPathVectors << endl;
        return;
    }
    if (!writerDetection.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для детекции: " << outputPathDetection << endl;
        return;
    }

    ofstream timeLogFile(timeLogPath);
    if (!timeLogFile.is_open()) {
        cerr << "Ошибка: не удалось открыть файл для лога времени: " << timeLogPath << endl;
        return;
    }

    Mat prevGray, nextGray, flow, colorFlow, vectorFlow, detectionFrame;
    int frameCount = 0;

    // Определяем область интереса (ROI) - прямоугольник впереди автомобиля
    // Например, нижняя половина кадра или центральная часть
    Rect roi(0, frameSize.height / 2, frameSize.width, frameSize.height / 2); // Пример: нижняя половина

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Преобразуем текущий кадр в полутоновый
        cvtColor(frame, nextGray, COLOR_BGR2GRAY);

        auto start = chrono::high_resolution_clock::now();

        // 1. Вычисляем плотный оптический поток, если есть предыдущий кадр
        if (!prevGray.empty()) {
            calcOpticalFlowFarneback(prevGray, nextGray, flow, pyrScale, levels, winsize, iterations, polyN, polySigma, flags);
        }
        else {
            // Для первого кадра создаем пустой поток
            flow = Mat::zeros(nextGray.size(), CV_32FC2);
        }

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0; // в мс

        // --- Обработка данных по области интереса ---
        double minMag = numeric_limits<double>::max();
        double maxMag = 0.0;
        double sumMag = 0.0;
        int countMag = 0;
        int aboveThreshold = 0;
        int belowThreshold = 0;

        // Пороговое значение (например, 20% от диапазона)
        double thresholdPercent = 0.20; // Можно изменять в экспериментах

        double range = 0.0;

        // Если есть поток, вычисляем статистику по ROI
        if (!flow.empty()) {
            for (int y = roi.y; y < roi.y + roi.height; ++y) {
                for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    float fx = flow.at<Vec2f>(y, x)[0];
                    float fy = flow.at<Vec2f>(y, x)[1];
                    double magnitude = sqrt(fx * fx + fy * fy);

                    if (magnitude < minMag) minMag = magnitude;
                    if (magnitude > maxMag) maxMag = magnitude;
                    sumMag += magnitude;
                    countMag++;
                }
            }

            // Вычисляем среднее
            double avgMag = (countMag > 0) ? sumMag / countMag : 0.0;
            range = maxMag - minMag;

            // Проверяем каждый пиксель в ROI на отклонение
            for (int y = roi.y; y < roi.y + roi.height; ++y) {
                for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    float fx = flow.at<Vec2f>(y, x)[0];
                    float fy = flow.at<Vec2f>(y, x)[1];
                    double magnitude = sqrt(fx * fx + fy * fy);

                    double deviation = abs(magnitude - avgMag);
                    if (range > 0 && deviation > thresholdPercent * range) {
                        if (magnitude > avgMag) {
                            aboveThreshold++;
                        }
                        else {
                            belowThreshold++;
                        }
                    }
                }
            }
        }

        // --- Запись статистики в лог-файл ---
        timeLogFile << frameCount << "\t" << duration << "\t"
            << minMag << "\t" << maxMag << "\t" << (countMag > 0 ? sumMag / countMag : 0.0) << "\t"
            << belowThreshold << "\t" << aboveThreshold << std::endl;

        // --- Генерация выходных изображений ---

        // 4. Цветное представление потока
        colorFlow = drawOpticalFlowColor(flow);
        writerColor.write(colorFlow);

        // 5. Векторное представление потока
        vectorFlow = drawOpticalFlowVectors(flow, 20); // Шаг 20 пикселов
        writerVectors.write(vectorFlow);

        // 6 & 7. Выделение движущихся объектов в области интереса
        detectionFrame = frame.clone(); // Копируем исходный кадр

        // Определяем область интереса (ROI) - прямоугольник впереди автомобиля
        // Поднимаем на 400 пикселей по сравнению с нижней половиной
        int roiY = max(0, frameSize.height / 2 - 400); // Верхняя граница ROI
        int roiHeight = min(frameSize.height / 2, frameSize.height - roiY); // Высота ROI (не больше половины и не выходит за пределы кадра)
        Rect roi(0, roiY, frameSize.width, roiHeight);

        // Выделяем пиксели, превышающие порог, ярко-красным цветом
        if (!flow.empty()) {
            for (int y = roi.y; y < roi.y + roi.height; ++y) {
                for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    float fx = flow.at<Vec2f>(y, x)[0];
                    float fy = flow.at<Vec2f>(y, x)[1];
                    double magnitude = sqrt(fx * fx + fy * fy);

                    double deviation = abs(magnitude - (countMag > 0 ? sumMag / countMag : 0.0));
                    if (range > 0 && deviation > thresholdPercent * range) {
                        detectionFrame.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // Ярко-красный
                    }
                }
            }
        }

        writerDetection.write(detectionFrame);

        // Подготовка к следующей итерации
        prevGray = nextGray.clone();
        frameCount++;

        if (frameCount % 30 == 0) { // Прогресс каждые 30 кадров
            std::cout << "Обработано кадров: " << frameCount << " из " << totalFrames << endl;
        }
    }

    cap.release();
    writerColor.release();
    writerVectors.release();
    writerDetection.release();
    timeLogFile.close();

    std::cout << "Видео " << inputPath << " обработано с параметрами:" << endl;
    std::cout << "  pyrScale=" << pyrScale << ", levels=" << levels << ", winsize=" << winsize
        << ", iterations=" << iterations << ", polyN=" << polyN << ", polySigma=" << polySigma << endl;
    std::cout << "Выходные файлы: " << outputPathColor << ", " << outputPathVectors << ", " << outputPathDetection << endl;
    std::cout << "Лог времени: " << timeLogPath << endl;
}

int main(int argc, char* argv[]) {
    // Путь к входному видео
    string videoPath = "C:/Users/bugro/Videos/country_ride.mp4";

    // Обработка с разными наборами параметров
    // Набор 1
    processVideo(videoPath,
        "video1_color_flow_set1.avi",
        "video1_vectors_flow_set1.avi",
        "video1_detection_set1.avi",
        "video1_timing_log_set1.txt",
        0.5, 3, 15, 3, 5, 1.2, 0);

    // Набор 2
    processVideo(videoPath,
        "video1_color_flow_set2.avi",
        "video1_vectors_flow_set2.avi",
        "video1_detection_set2.avi",
        "video1_timing_log_set2.txt",
        0.5, 1, 15, 3, 5, 1.2, 0);

    // Набор 3
    processVideo(videoPath,
        "video1_color_flow_set3.avi",
        "video1_vectors_flow_set3.avi",
        "video1_detection_set3.avi",
        "video1_timing_log_set3.txt",
        0.5, 1, 15, 1, 5, 1.2, 0);

    std::cout << "Обработка завершена. Проверьте выходные файлы." << endl;

    return 0;
}