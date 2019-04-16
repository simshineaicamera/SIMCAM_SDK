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

//A shared memory pointer that controls whether video is recorded,whether recognition box is displayed and whether play audio. 
//Users can not create other shared memory pointer.
stMemory* viraddr;

//A variable containing preprocess parameters of Detect_Graph, stores values from config.txt
static CNN_Config_t gCNNparam;
static Server_Info server_info;
int classID;
RectEx_t Rect;
//color of recognition box.
YUVColor_t YUVColor_Orange={173,186,30};

// A function reads values from config.txt and stores them in a structure.
int readConfig(CNN_Config_t* config, Server_Info* info){

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
    
    info->if_send=cJSON_GetObjectItem(json, "if_send")->valueint;
    info->server_ip=cJSON_GetObjectItem(json,"server_IP")->valuestring;
    info->port=cJSON_GetObjectItem(json,"port")->valueint;

    return 0;
}
void socket_send(int classid,RectEx_t rectt){
    char* server_ip =server_info.server_ip;
    int socketfd;    
    struct sockaddr_in sockaddr;    
    socketfd = socket(AF_INET, SOCK_STREAM, 0);    
    memset(&sockaddr, 0, sizeof(sockaddr));    
    sockaddr.sin_family = AF_INET;    
    sockaddr.sin_port = htons(server_info.port);   
    inet_pton(AF_INET, server_ip, &sockaddr.sin_addr);    
    if ((connect(socketfd, (struct sockaddr*)&sockaddr, sizeof(sockaddr))) <0){            
        printf("server connect error\n");
        return;        
    }
   // printf(" connect succeed \n");
   struct timeb timer_msec;
    long long int timestamp_msec;
    if(!ftime(&timer_msec)){
        timestamp_msec=((long long int)timer_msec.time)*1000ll + (long long int)timer_msec.millitm;
    }
    else
    {
        timestamp_msec=-1;
    }
   
    char miio_send[1024];
    memset(miio_send, 0, sizeof(miio_send));
    // 
    sprintf(miio_send,"GET http://%s:%d/simcam/event?modeltype=%d&pointA_X=%f&pointA_Y=%f&pointB_X=%f&pointB_Y=%f&timestamp=%lld\r\n\r\nCache-Control: no-cache",
    server_info.server_ip, server_info.port, classid,rectt.rect.x1,rectt.rect.y1,rectt.rect.x2,rectt.rect.y2,timestamp_msec);
    // sprintf(miio_send,"GET /simcam/event?modeltype=1&pointA_X=1&pointA_Y=2&pointB_X=3&pointB_Y=4&timestamp=1234567890111 HTTP/1.1\r\n\
    //         Host: 118.190.201.26:8001\r\n\
    //         Cache-Control: no-cache");
        if ((send(socketfd, miio_send, strlen(miio_send), 0)) < 0) {
            printf("tcp send fail to server\n");
        }

    close(socketfd);
}

// A function parses the output data of CNN network, gCmd is a globle value and the original output data
void cmdHandler(){
    //create a structure to copy the value from gCmd at this moment
    cmd_t _ptr;
    memset(&_ptr, 0, sizeof(cmd_t));
    memcpy(&_ptr, &gCmd, sizeof(cmd_t));
   
    //the first value of data[1024] is the number of recognition boxes
    int num_box =  _ptr.data[0];


    //if the number of boxes is not in range of 0~4, there may be a mistake.
    if(num_box<0 || num_box>4) return;

    for(int i = 0; i < num_box; i++){
        // create a structure to store the value of coordinates 
        
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
        classID = _ptr.data[(i + 1) * 7 + 1];

        //print message of the result,including class ID and coordinates of recognition box.
        printf("DetectionModelResult: class: %d, x1: %f, y1: %f, x2: %f, y2: %f\n",classID,_ptr.data[10],_ptr.data[11],_ptr.data[12],_ptr.data[13]);

        //the ID number is 12, which means that the object is identified as a dog.
        // if(12 == classID){
        //     sprintf(localdata.path_audio,"dog");     //play a audio to remind you it's a dog. "dog.pcm" should be in folder /mnt/DCIM/voice.
        // }
   
        //copy all the value of localdata to the shared memory.
        memcpy(viraddr,&localdata,sizeof(stMemory));
  //  }
    }
     // Print classification result ----->>>>>
    if(gCNNparam.cnn[1].model_have){
        float res1=_ptr.data[70+7];
        float res2=_ptr.data[70+8];
        if(res1>=1.0||res1<=0)return;
        if(res2>=1.0||res2<=0)return;
        printf("FirstClassificationModelResult, class0: %f\n", res1);
        printf("FirstClassificationModelResult, class1: %f\n", res2);
        }
    // Print classification result ----->>>>>
    if(gCNNparam.cnn[2].model_have){
        float res1=_ptr.data[326+7];
        float res2=_ptr.data[326+8];
        if(res1>=1.0||res1<=0)return;
        if(res2>=1.0||res2<=0)return;
        printf("SecondClassificationModelResult, class0: %f\n", res1);
        printf("SecondClassificationModelResult, class1: %f\n", res2);
        }
    if(server_info.if_send){
      socket_send(classID, Rect);
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


void rebootAlg(CNN_Config_t* param, Server_Info* info){

//send the server process program of Movidius through SPI.
        sendApp("Detect_Server_Process");
//after sent the server process program,enable SPI.
        openSpi();
//call the function of readConfig to read parameters from config.txt.
        readConfig(param, info);
        
   
    struct timeb timer_msec;
    long long int timestamp_msec;
    if(!ftime(&timer_msec)){
        timestamp_msec=((long long int)timer_msec.time)*1000ll + (long long int)timer_msec.millitm;
    }
    else
    {
        timestamp_msec=-1;
    }
    
    printf("if_send:%d, ip:%s, port:%d, time:%lld \n", server_info.if_send,server_info.server_ip, server_info.port, timestamp_msec);
//after read the parameters,send them to Movidius through SPI.
        sendCfg((uint8_t*)param,sizeof(CNN_Config_t));
//send the model file to Movidius through SPI.
        sendBlob("person_face"); // detection model
        if(gCNNparam.cnn[1].model_have){
            sendBlob("gender");  // gender classification model
        }
         if(gCNNparam.cnn[2].model_have){
            sendBlob("emotion"); // emotion classification model
        }
        
}
void set_time(){
    system("ntpd -p us.ntp.org.cn -qNn");
   // return 0;
}
int main(int argc, char *argv[])
{
//init the whole system,including SPI and serial port.
    initSystem();
    set_time();
//reset Movidius.
    reset2450();

    rebootAlg(&gCNNparam, &server_info);
   
    startSpiServer();
    
    while(1);
}
