#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "rectify.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    int task_number;
    if (argc < 2 || string(argv[1]) == "task1") {
        task_number = 1;
    } else if (string(argv[1]) == "task2") {
        task_number = 2;
    } else if (string(argv[1]) == "task3") {
        task_number = 3;
    } else{
        cout<<"please input task1, task2 or task3!"<<endl;
        return -1;
    }
    const string input_video = "../videos/stereo.avi";
    const string output_video = "../videos/";

    VideoCapture cap(input_video);
    VideoWriter video_writer;
    if (task_number == 1) {
        video_writer = VideoWriter(output_video + "task1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                   40, Size(720, 288));
    } else if (task_number == 2) {
        video_writer = VideoWriter(output_video + "task2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                   40, Size(720, 288));
    } else if (task_number == 3) {
        video_writer = VideoWriter(output_video + "task3.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                   20, Size(360, 288));
    }

    if (!cap.isOpened()) {
        cout << "opencv stereo.avi failed!" << endl;
        return -1;
    }

    Rectify rec;
    while (1) {
        Mat frame;
        shared_ptr<Mat> out_frame = nullptr;
        cap.read(frame);
        if (frame.empty()) {
            cout << "finished!" << endl;
            break;
        }
        if (task_number == 1) {
            out_frame = rec.GetAlignImage(frame);
        } else if (task_number == 2) {
            out_frame = rec.GetMatchedImage(frame);
        } else if (task_number == 3) {
            out_frame = rec.GetDisparityImage(frame);
        }
        imshow("origin", frame);

        if (out_frame) {
            video_writer.write(*out_frame);
            imshow("rectify", *out_frame);
        }

        waitKey(25);
    }
    cap.release();
    video_writer.release();
    return 0;
}
