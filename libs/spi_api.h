#ifndef SPI_API_H
#define SPI_API_H

#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <stdint.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define TX_LEN 256
#define BUFF_LEN 4096

static const char* device = "/dev/spidev0.0";
static uint8_t mode = 3;
static uint8_t bits = 8;
static uint32_t speed = 12000000;
static uint16_t delay = 0;

typedef struct{
    char cmd;
    int bytes;
    float data[1024];
}cmd_t;

extern cmd_t gCmd;

void openSpi(void);

void closeSpi(void);

int readCmd( void (*cmdHandler)() );

void writeCmd(uint8_t* buf, uint32_t sz);

void sendBlob(const char* model_path);

void sendApp(const char* app_path);

void sendCfg(const uint8_t* ssd,uint32_t len);

#endif
