#include <iostream>
#include <time.h>
#include <string>
#include <vector>
#include <fstream>
extern "C"{
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <errno.h>
    #include <dirent.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
}

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "../../../libs/sqlite_db.h"
sqlite3 *pdb;
using namespace std; 
using namespace cv::dnn;
using namespace cv;

size_t inWidth_lcnn = 128;
size_t inHeight_lcnn = 128;
double inScaleFactor_lcnn = 1/256.0f;
const float meanVal_lcnn = 0;

size_t inWidth_ssd = 300;
size_t inHeight_ssd = 300;
double inScaleFactor_ssd = 1/127.5f;
const float meanVal_ssd = 127.5;
#define Display_width 720
#define Display_height 720

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}

void getAllFiles(string path, vector<string>& files)
{
    DIR *dir;
    struct dirent *ptr;
    if((dir=opendir(path.c_str()))==NULL){
        perror("Open dri error...");
        exit(1);
    }
    while((ptr=readdir(dir))!=NULL){
        if(strcmp(ptr->d_name,".")==0||strcmp(ptr->d_name,"..")==0)
            continue;
        else if(ptr->d_type==8)//file
            files.push_back(path+"/"+ptr->d_name);
        else if(ptr->d_type==10)//link file
            continue;
        else if(ptr->d_type==4){
            //files.push_back(ptr->d_name);//dir
            getAllFiles(path+"/"+ptr->d_name,files);
        }
    }
    closedir(dir);
}

void Min_Area(vector<cv::Rect> &v_face, cv::Rect &face)//max
{
    if(v_face.size()<1)return;
    if(v_face.size() == 1)
    {
        face=v_face[0];
        return;
    }
    int Min_area = v_face[0].width*v_face[0].height, index = 0;
    for(int i = 1;i<v_face.size();i++)
    {
        if(v_face[i].width*v_face[i].height < Min_area)
        {
            index = i;
            Min_area = v_face[i].width*v_face[i].height;
        }
    }
    face = v_face[index];
}

#define GoogleNet_Dim 128
void hist_equa(unsigned char * in_gray,unsigned char * out_gray)
{
    //直方图
	unsigned int hist[256] = {0},i,j;
    unsigned char temp[GoogleNet_Dim*GoogleNet_Dim];
    float size = GoogleNet_Dim*GoogleNet_Dim;
	for (i=0; i<GoogleNet_Dim; i++)
	{
		for (j=0; j<GoogleNet_Dim; j++)
		{
			temp[i*GoogleNet_Dim+j] = in_gray[i*GoogleNet_Dim+j];
			hist[temp[i*GoogleNet_Dim+j]]++;
		}
	}
    //归一化直方图
	float histPDF[256] = {0};
	for (i=0; i<256; i++)
	{
		histPDF[i]=(float)hist[i]/size;
	}
	//累积直方图
	float histCDF[256] = {0};
    histCDF[0] = histPDF[0];
	for (i=1; i<256; i++)
	{
		histCDF[i] = histCDF[i-1] + histPDF[i];
	}
    //直方图均衡化,映射
	int histEQU[256] = {0};
	for (i=0; i<256; i++)
	{
		histEQU[i] = (int)(255.0 * histCDF[i] + 0.5);
	}
	for (i=0; i<GoogleNet_Dim; i++)
	{
		for (j=0; j<GoogleNet_Dim; j++)
		{
			out_gray[i*GoogleNet_Dim+j] = histEQU[temp[i*GoogleNet_Dim+j]];
		}
	}
    
}

int main(int argc, char** argv) 
{   
    if(argc<2){
        cout<<"USAGE:./main face_images/"<<endl;
        exit(-1);
    }
    char * filePath = argv[1];
    if(filePath[strlen(filePath)-1]=='/')filePath[strlen(filePath)-1]=0;
    vector<string> files;
    getAllFiles(filePath, files);
   remove("faces.db");
   int fd = create_db(pdb);

   fd = sqlite3_open("faces.db", &pdb);
        if (fd != SQLITE_OK)
    {
        printf("can not open database!\n");
        sqlite3_close(pdb);
        return -1;
    }
    // 使用时检查输入的参数向量是否为要求的6个，如果不是，打印使用说明
    // 这里可以根据个人需要更改，是否需要均值文件等...
    string modelTxt1   = "../../models/lcnn/light_cnn_deploy.prototxt";
    string modelBin1 = "../../models/lcnn/light_cnn.caffemodel";

    //! [Initialize network]
    dnn::Net lcnn_net = readNetFromCaffe(modelTxt1, modelBin1);
    
    if (lcnn_net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        exit(-1);
    }

    string modelTxt2 = "../../models/person_face/deploy.prototxt";
    string modelBin2 = "../../models/person_face/person_face.caffemodel";

    //! [Initialize network]
    dnn::Net ssd_net = readNetFromCaffe(modelTxt2, modelBin2);
    

    if (ssd_net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        exit(-1);
    }

    for (int i = 0; i<files.size(); i++)
    {
        Mat ssd_frame,ssd_img_resize,ssd_display;
        
        ssd_frame=imread(files[i]);
        if (ssd_frame.empty())
        {
            cout << files[i] <<" image is empty" << endl;
            continue;
        }
        resize(ssd_frame, ssd_img_resize, Size(inWidth_ssd,inHeight_ssd));
        resize(ssd_frame, ssd_display, Size(Display_width,Display_height));
        
        //! [Prepare blob]
        Mat inputBlob = blobFromImage(ssd_img_resize, inScaleFactor_ssd,
                                      Size(inWidth_ssd, inHeight_ssd), Scalar(meanVal_ssd, meanVal_ssd, meanVal_ssd), false); //Convert Mat to batch of images
        
        //! [Set input blob]
        ssd_net.setInput(inputBlob, "data"); //set the network input
        
        //! [Make forward pass]
        Mat detection = ssd_net.forward(); //compute output
        
        vector<cv::Rect> faces;
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > 0.3)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));
                printf("class:%d,conf:%f,x1:%f,y1:%f,x2:%f,y2:%f\n",objectClass,confidence,detectionMat.at<float>(i, 3),detectionMat.at<float>(i, 4),
                detectionMat.at<float>(i, 5),detectionMat.at<float>(i, 6));

                if(objectClass == 3 && confidence > 0.5)
                {
                    int left = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 3))) * ssd_frame.cols);
                    int top = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 4))) * ssd_frame.rows);
                    int right = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 5))) * ssd_frame.cols);
                    int bottom = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 6))) * ssd_frame.rows);
                    faces.push_back(cv::Rect(Point(left, top),Point(right, bottom)));
                }
                
                int left = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 3))) * ssd_display.cols);
                int top = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 4))) * ssd_display.rows);
                int right = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 5))) * ssd_display.cols);
                int bottom = static_cast<int>(max(0.01,min(0.99,detectionMat.at<float>(i, 6))) * ssd_display.rows);
                rectangle(ssd_display, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
                
            }
        }
        imshow("detections", ssd_display);
        //waitKey(50);
        if(faces.size() < 1)
        {
            cout << " warning " << files[i] << endl;
            continue;
        }
        cv::Rect face_max;
        Min_Area(faces,face_max);
        Mat lcnn_face = ssd_frame(face_max);
        //imshow("lcnn_face",lcnn_face);
        cv::waitKey(10);
        //imwrite("temp.jpg",lcnn_face);

        Mat img_resize,lcnn_image;
        resize(lcnn_face, img_resize, Size(inWidth_lcnn,inHeight_lcnn));
        cvtColor(img_resize,lcnn_image,cv::COLOR_BGR2GRAY);
        //imshow("lcnn_face_gray",lcnn_image);
        //Mat hist_lcnn_img=lcnn_image.clone();
        //hist_equa(lcnn_image.data,hist_lcnn_img.data);
        //! [Prepare blob]
        Mat inputBlob2 = blobFromImage(lcnn_image, inScaleFactor_lcnn,
                                        Size(inWidth_lcnn, inHeight_lcnn), Scalar(meanVal_lcnn), false); //Convert Mat to batch of images
        //! [Prepare blob]
        //! [Set input blob]
        lcnn_net.setInput(inputBlob2, "data"); //set the network input
        //! [Set input blob
        //! [Make forward pass]
        Mat detection2 = lcnn_net.forward(); //compute output, size = 1,256,1,1
        float feature[face_feature_lenght],*fea = detection2.ptr<float>();
        memcpy(feature,fea,face_feature_lenght*sizeof(float));
        
        vector<string> v_path;
        SplitString(files[i],v_path,"/");
        // printf("\n features:");
        // for(int kk=0;kk<256;kk++){
        // printf(" %f ", feature[kk]);
        // }
      //  printf("\n--------->>>>>>>>>>>>\n");
        insert_db(pdb, v_path[v_path.size()-2].c_str(), feature);
    }

    
    sqlite3_close(pdb);
    return 0;
    //Mat detectionMat(detection.size[0], detection.size[1], CV_32F, detection.ptr<float>());
}
