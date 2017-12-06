/**
 * Cika Desela
 * 00000011818
 * IMG PROCESSING PROJECT
 * Real-time shape detection & identification
 */


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;


void setLabel(cv::Mat& im, const std::string label, std::vector <cv:: Point>& contour)
{
    int fontface= cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseLine=0;

    cv::Size text=cv::getTextSize(label, fontface,scale,thickness, &baseLine);
    cv::Rect r= cv:: boundingRect(contour);

    cv::Point pt(r.x+((r.width-text.width)/2), r.y +((r.height+text.height)/2));
    cv::rectangle(im, pt +cv::Point(0,baseLine),pt+cv::Point (text.width, -text.height),CV_RGB(255,255,255),CV_FILLED);
    cv::putText(im, label, pt, fontface,scale,CV_RGB(0,0,0), thickness,8);
}

static double angle (cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 =pt1.x-pt0.x;
    double dy1 =pt1.y-pt0.y;
    double dx2 =pt2.x-pt0.x;
    double dy2 =pt2.y-pt0.y;
    return (dx1*dx2+dy1*dy2)/sqrt((dx1*dx1+dy1*dy1)*(dx2*dx2+dy2*dy2)+1e-10);
}

int main()
{
    Mat src;
    Mat gray;
    Mat bw;
    Mat dst;
    std::vector<std::vector<cv::Point> >contours;
    std::vector<cv::Point> approx;
    vector<Vec4i> hierarchy;

    RNG rng(12345);



    VideoCapture capture(0);
    int q;
    while(cvWaitKey(30) != 'q')
    {
        capture>> src;
        if(true){
            cv::cvtColor(src,gray,CV_BGR2GRAY);

            ///reduce noise to avoid false detection
            blur(gray,bw,Size(3,3));
            Canny(gray,bw,255,255/3,3);
            imshow("Scan",bw);

            findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE,Point(0,0));
            src.copyTo(dst);

            ///new again
            vector<Vec3f> circles;
            HoughCircles(gray,circles,CV_HOUGH_GRADIENT,1,gray.rows/8,200,100,0,0);
            /// Draw the detected circles
            for( size_t i = 0; i < circles.size(); i++ )
            {
                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                // circle center
                circle( dst, center, 3, Scalar(0,255,0), -1, 8, 0 );
                // circle outline
                circle( dst, center, radius, Scalar(0,0,255), 3, 8, 0 );
            }

            ///new
            vector<vector<Point> >contours_poly(contours.size());
            vector<Rect> boundRect (contours.size());
            vector<Point2f> center (contours.size());
            vector<float> rad(contours.size());
            vector<vector<Point> >hull(contours.size());

                for( int i = 0; i < contours.size(); i++ )
                {
                    approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );
                    boundRect[i] = boundingRect( Mat(contours_poly[i]) );
                    minEnclosingCircle( (Mat)contours_poly[i], center[i], rad[i] );
                    convexHull( Mat(contours[i]), hull[i], false );

                    if( contours_poly[i].size()>15) // Check for corner
                        drawContours( dst, contours_poly, i, Scalar(0,255,0), 2, 8, vector<Vec4i>(), 0, Point() ); // True object with green color
                    //else
                    //  drawContours( src, contours_poly, i, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point() ); // false object with blue color
                    //drawContours( src, hull, i, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point() );
                    // rectangle( src, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
                    //circle( src, center[i], (int)rad[i], Scalar(0,0,255), 2, 8, 0 );
                }

            ///new ends

            /* MOVED
            findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE,Point(0,0));
            src.copyTo(dst);
            */

            Mat drawing = Mat::zeros( gray.size(), CV_8UC3 );
            for (int i=0; i<contours.size();i++)
            {
                //coloring contours
                Scalar color=Scalar (rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255) );
                drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );

                cv::approxPolyDP(Mat (contours[i]), approx, cv::arcLength(Mat(contours[i]),true)*0.02,true);
                    if(std::fabs(cv::contourArea(contours[i]))<100 || !cv::isContourConvex(approx))
                        continue;

                    if(approx.size()==3){
                        setLabel(dst, "Triangle",contours[i]);
                    }
                    else if (approx.size()>= 4 && approx.size()<=6)
                    {
                    int vtc= approx.size();
                    std::vector<double> cos;
                    for (int j=2; j< vtc+1; j++)

                        cos.push_back(angle(approx[j%vtc], approx [j-2], approx[j-1]));

                        std::sort(cos.begin(), cos.end());

                        double mincos = cos.front();
                        double maxcos = cos.back();

                        if(vtc==4)
                                setLabel(dst, "Square", contours[i]);
                        else if (vtc==5)
                                setLabel (dst, "Pentagon", contours[i]);
                        else if (vtc==6)
                            setLabel (dst, "Hexagon", contours[i]);
                        else if (vtc==7)
                           setLabel(dst, "Heptagon", contours[i]);
                        else if (vtc==8)
                            setLabel(dst, "Octagon", contours[i]);

                    }
                    else
                    {
                        double area = cv::contourArea(contours[i]);
                        Rect r= cv::boundingRect(contours[i]);
                        int radius = r.width/2;

                        if( abs(1- (area/ CV_PI *(radius*radius))<= 0.1 &&
                            abs(1- ((double)r.width/r.height)))<= 0.1)
                            setLabel(dst, "Circle", contours[i]);
                            cout<<"Circle contours area: "<<area<<endl;
                    }




                }

                imshow("Cam",dst);
                imshow("Colored Shapes (contours)", drawing);
            }

            else{
                break;
            }

        }
    return 0;
}
