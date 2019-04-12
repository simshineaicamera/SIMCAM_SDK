#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <signal.h>
#include <pthread.h>
#include <signal.h>
#include "../../libs/SIMCAM_lib.h"

//-------->>>>>>>>>>>
#include "../../libs/sqlite_db.h"
#include "math.h"
#include <iostream>

sqlite3 *pdb;
using namespace std;

float GetFaceFeatureSimility(float  *feature1,float  *feature2);
int Top_one(vector<string> names,vector<float> simi);

vector<float> v_simility;
    
vector<string> v_name;
///->>>>>>>>>>>>>>>>>>
//A shared memory pointer that controls whether video is recorded,whether recognition box is displayed and whether play audio. 
//Users can not create other shared memory pointer.
stMemory* viraddr;

//A variable containing preprocess parameters of Detect_Graph, stores values from config.txt
static CNN_Config_t gCNNparam;

//color of recognition box.
YUVColor_t YUVColor_Orange={173,186,30};

// A function reads values from config.txt and stores them in a structure.
int readConfig(CNN_Config_t* config){

    //open the file of config.txt in SD-Card.
    FILE* fd = fopen("/mnt/DCIM/config.txt", "r");
    if(fd==NULL){
        printf("open /mnt/DCIM/config.txt err\n");
        return -1;
    } 

    //Get the content of config.txt
    char tmp[4096]={0};
    fread(tmp,1,4096,fd);
    fseek(fd,0,SEEK_SET);

    //Use JSON to parses the content from config.txt and stores the parameters in a structure.
    cJSON *json,*arrayItem,*item,*object;
	json = cJSON_Parse(tmp);
	arrayItem=cJSON_GetObjectItem(json,"model");
	int size=cJSON_GetArraySize(arrayItem);
    config->NumOfModel = size;
	for(int i=0;i<size;i++){
        object=cJSON_GetArrayItem(arrayItem,i);
        config->cnn[i].model_have = cJSON_GetObjectItem(object,"model_have")->valueint;
        config->cnn[i].input_width = cJSON_GetObjectItem(object,"input_width")->valueint;
        config->cnn[i].input_height = cJSON_GetObjectItem(object,"input_height")->valueint;
        config->cnn[i].shave_num = cJSON_GetObjectItem(object,"shave_num")->valueint;
        config->cnn[i].input_color = cJSON_GetObjectItem(object,"input_color")->valueint;
        config->cnn[i].mean0 = cJSON_GetObjectItem(object,"mean0")->valuedouble;
        config->cnn[i].mean1 = cJSON_GetObjectItem(object,"mean1")->valuedouble;
        config->cnn[i].mean2 = cJSON_GetObjectItem(object,"mean2")->valuedouble;
        config->cnn[i].std = cJSON_GetObjectItem(object,"std")->valuedouble;
        config->cnn[i].label = cJSON_GetObjectItem(object,"label")->valueint;
        config->cnn[i].conf_thresh = cJSON_GetObjectItem(object,"conf_thresh")->valuedouble;
    }
    return 0;
}

// A function parses the output data of CNN network, gCmd is a globle value and the original output data
void cmdHandler(){
    //create a structure to copy the value from gCmd at this moment
    cmd_t _ptr;
    memset(&_ptr, 0, sizeof(cmd_t));
    memcpy(&_ptr, &gCmd, sizeof(cmd_t));
   int face_index = 0;
    //the first value of data[1024] is the number of recognition boxes
    int num_box =  _ptr.data[0];


    //if the number of boxes is not in range of 0~4, there may be a mistake.
    if(num_box<0 || num_box>4) return;

    for(int i = 0; i < num_box; i++){
        v_name.clear();
        v_simility.clear();
        // create a structure to store the value of coordinates 
        RectEx_t Rect;
            //
            Rect.rect.x1 = _ptr.data[(i + 1) * 7 + 3];
            Rect.rect.y1 = _ptr.data[(i + 1) * 7 + 4];
            Rect.rect.x2 = _ptr.data[(i + 1) * 7 + 5];
            Rect.rect.y2 = _ptr.data[(i + 1) * 7 + 6];
            
            //the value of each coordinate is in range of 0~1, if thet are not in this range ,there may be a mistake.
            if ((Rect.rect.x1 < 0) || (Rect.rect.x1 > 1))   return;
            if ((Rect.rect.y1 < 0) || (Rect.rect.y1 > 1))   return;
            if ((Rect.rect.x2 < 0) || (Rect.rect.x2 > 1))   return;
            if ((Rect.rect.y2 < 0) || (Rect.rect.y2 > 1))   return;

        //Create a structure to set whether video is recorded, whether voice is played, and whether recognition boxes are displayed.
        stMemory localdata;
        memset(&localdata,0,sizeof(stMemory));

        //set the color of recognition box
        localdata.rect_data.color = YUVColor_Orange;
        //set the coordinates of recognition box
        localdata.rect_data.rect = Rect.rect;
        //display the recognition box
        localdata.rect_data.if_show = 1;
        //No video recording
        localdata.if_record = 0;

        //get the probility from data[1024]
        float probility = _ptr.data[(i + 1) * 7 + 2];
        //get the threshold from config.txt
        float con_thresh = gCNNparam.cnn[0].conf_thresh;
        //if the probility is too small,there may be a mistake.
        if(probility<con_thresh) return;
        
        //get the class ID of detected object,the corresponding category of ID can be viewed in labelmap.prototxt
        int classID = _ptr.data[(i + 1) * 7 + 1];

        //print message of the result,including class ID and coordinates of recognition box.
        printf("DetectionModelResult: class: %d, x1: %f, y1: %f, x2: %f, y2: %f\n",classID,_ptr.data[10],_ptr.data[11],_ptr.data[12],_ptr.data[13]);
        
       
   
        //copy all the value of localdata to the shared memory.
        memcpy(viraddr,&localdata,sizeof(stMemory));
        if(classID!=3)continue;
        
        float feature[256];
        float *feature_p;
        if(face_index>2)face_index=0;
        for(int j=0;j<256;j++){
           feature[j]=_ptr.data[77+260*face_index+j];
         //  printf("%f ",feature[j]);
        }
        feature_p=feature;
       // printf("%d \n", face_index);
        face_index++;
        query_db(pdb, GetFaceFeatureSimility, feature_p,v_name,v_simility,0.1);

        if(v_name.size()<1)continue;
            
           int index =Top_one(v_name,v_simility);
          // printf("index:%d,simi:%f\n",index,v_simility[index]);
   }
    }
   
  
    
   

//create a thread to read data from SPI 
void* threadSpi(void *arg){
	int Cnt = 0;
    while (1) {
	    if(Cnt++%30==0){
            printf("No detection, spi loop>>>>>>>>>>>>>>>>>>>>>>>>\n");
            memset(viraddr,0,sizeof(stMemory));
        }
        //readCmd() will get all data from SPI and verify the validity of the data,only valid data will be further processed in cmdHandler().
        readCmd(cmdHandler);
        Cnt++;
        usleep(100*1000);
    }
    pthread_exit(0);
}

//start to run the thread created above.
void startSpiServer(){
    pthread_t spi_ID;
    pthread_create(&spi_ID, NULL, &threadSpi, NULL);
    pthread_detach(spi_ID);
}


void rebootAlg(CNN_Config_t* param){

//send the server process program of Movidius through SPI.
        sendApp("Detect_Server_Process");
//after sent the server process program,enable SPI.
        openSpi();
//call the function of readConfig to read parameters from config.txt.
        readConfig(param);
//after read the parameters,send them to Movidius through SPI.
        sendCfg((uint8_t*)param,sizeof(CNN_Config_t));
//send the model file to Movidius through SPI.
        sendBlob("person_face"); // detection model
        if(gCNNparam.cnn[1].model_have){
            sendBlob("lcnn");  // for face recognition
        }
        
        
}

int main(int argc, char *argv[])
{
//init the whole system,including SPI and serial port.
    initSystem();
//reset Movidius.
    reset2450();
    rebootAlg(&gCNNparam);
    
    startSpiServer();
    int fd;
    fd = sqlite3_open("./faces.db", &pdb);
    if (fd)
    {
        printf("can not open database!\n");
        sqlite3_close(pdb);
        return -1;
    }
    while(1);
}
float GetFaceFeatureSimility(float  *feature1,float  *feature2)
{
    if(feature1==NULL||feature2==NULL) 
    {    
       //  printf("In the func Classifier::GetFaceFeatureSimility(),the input param feature1 or feature2 is err!\n");
         return 0;
    }
    float sumarrayA=0,sumarrayB=0;
    float cosine=0;
    float *temp_feature1=(float *)feature1;
    float *temp_feature2=(float *)feature2;
    int temp_len=256;
    
    for(int i=0;i<temp_len;i++)
    {
       // printf("num: %d fea1:%f,fea2:%f\n",i, temp_feature1[i],temp_feature2[i]);
        sumarrayA+=temp_feature1[i]*temp_feature1[i];
        sumarrayB+=temp_feature2[i]*temp_feature2[i];
        cosine+=temp_feature1[i]*temp_feature2[i];
    }
    sumarrayA=sqrt(sumarrayA);
    sumarrayB=sqrt(sumarrayB);
    if((sumarrayA-0<0.0001)||(sumarrayB-0<0.0001))
    {
        printf("In the func Recognition::GetFaceFeatureSimility(),the value of the input param feature1 or feature2 is 0!\n");
        return 0;
    }

    cosine =cosine/(sumarrayA*sumarrayB);
    return cosine;
}

int Top_one(vector<string> names,vector<float> simi)
{
    float max = -100;
    int i=0,index=0,len=names.size();

        for(i =0;i<len;i++)
        {
       printf("Supposition: %s similarity: %f \n",names[i].c_str(), simi[i]);
            if(simi[i]>=max)
            {
                max = simi[i];
                index=i;
            }
        }
    printf("Recognized as : %s \n",names[index].c_str());
    return index;
}


