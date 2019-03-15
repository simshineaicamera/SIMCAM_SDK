#ifndef __MOTOR_CTL_H__
#define __MOTOR_CTL_H__



int openSerial(char* device);

void closeSerial(int fd);

int initSerial(int fd, int speed,int flow,int bits,int stop,int parity);

int sendSerial(int fd, char *buf,int len);

int recvSerial(int fd, char *buf,int len);

int Wait4SendDone();

#endif
