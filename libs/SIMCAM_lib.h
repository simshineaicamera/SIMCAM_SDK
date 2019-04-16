#ifndef __OPEN_LIB_H
#define __OPEN_LIB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <signal.h>
#include <pthread.h>
#include "spi_api.h"
#include "serial_api.h"
#include "cJSON.h"
#include <signal.h>
#include <netdb.h>  
#include <net/if.h>  
#include <arpa/inet.h>  
#include <sys/ioctl.h>  
#include <sys/types.h>  
#include <sys/socket.h>
#include <sys/timeb.h>
#include <sys/time.h>

typedef struct{
    float x1;
    float y1;
    float x2;
    float y2;
}Rectf_t;

typedef struct{
    uint8_t Y;
    uint8_t U;
    uint8_t V;
}YUVColor_t;

typedef struct{
    Rectf_t rect;
    YUVColor_t color;
    int if_show; 
}RectEx_t;

typedef struct{
    int if_record;
    RectEx_t rect_data;
    char path_audio[64];
}stMemory;

typedef struct{    
    int model_have; 
    int input_width;    
    int input_height;    
    int shave_num;    
    int input_color;//0:gray 1:rgb    
    float mean0;    
    float mean1;    
    float mean2;    
    float std;
    int label;
    float conf_thresh;
}CNN_Param_t;

typedef struct{
    int NumOfModel;    
    CNN_Param_t cnn[3];
}CNN_Config_t;

typedef struct{
    int if_send;
    char*  server_ip;
    int port;
}Server_Info;

extern stMemory* viraddr;

void initSystem();
void reset2450();
//int readConfig(CNN_Config_t* config);

#endif