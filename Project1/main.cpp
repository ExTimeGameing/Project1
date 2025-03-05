#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "chrono"
#include "iostream"

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
    cv::Size resizedSize(400, 300);

    cv::VideoWriter writerStandart("C:/Users/bugro/Videos/output.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize, false);

    cv::VideoWriter writerResized("C:/Users/bugro/Videos/output2.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, resizedSize, false);

    double totalTimeStandartSize = 0;
    double totalTimeResizedSize  = 0;
    double totalResizeTime       = 0;
    double frameCount            = 0;


    cv::Mat frame, edges, gray, resizedFrame, resizedGray, resizedEdges;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        frameCount++;

        auto startOriginal = std::chrono::high_resolution_clock::now();

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);

        auto endOriginal = std::chrono::high_resolution_clock::now();
        totalTimeStandartSize += std::chrono::duration<double, std::milli>(endOriginal - startOriginal).count();

        writerStandart.write(edges);

        auto startResize = std::chrono::high_resolution_clock::now();

        cv::resize(frame, resizedFrame, resizedSize);

        auto endResize = std::chrono::high_resolution_clock::now();
        totalResizeTime += std::chrono::duration<double, std::milli>(endResize - startResize).count();

        auto startResized = std::chrono::high_resolution_clock::now();

        cv::cvtColor(resizedFrame, resizedGray, cv::COLOR_BGR2GRAY);
        cv::Canny(resizedGray, resizedEdges, 50, 150); // »спользуем те же параметры

        auto endResized = std::chrono::high_resolution_clock::now();
        totalTimeResizedSize += std::chrono::duration<double, std::milli>(endResized - startResized).count();

        writerResized.write(resizedEdges);

        // ѕоказываем кадры
        cv::imshow("Original", edges);
        cv::imshow("Resized", resizedEdges);

        if (cv::waitKey(33) >= 0) break;
    }

    setlocale(LC_ALL, "Russian");

    std::cout << "—реднее врем€ обработки ќ–»√»ЌјЋ№Ќќ√ќ кадра: "
        << totalTimeStandartSize / frameCount << " мс" << std::endl;

    std::cout << "—реднее врем€ обработки ”ћ≈Ќ№Ў≈ЌЌќ√ќ кадра: "
        << totalTimeResizedSize / frameCount << " мс" << std::endl;

    std::cout << "—реднее врем€ на уменьшение кадра: "
        << totalResizeTime / frameCount << " мс" << std::endl;

    cap.release();
    writerStandart.release();
    writerResized.release();
    cv::destroyAllWindows();

    return 0;
}