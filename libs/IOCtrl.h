#ifndef IOCTRL_H
#define IOCTRL_H

#define   GPIO_READ             0
#define   GPIO_WRITE            1
#define   KEY_PRESS             0
#define   KEY_LONG_PRESS        2
#define   POWER_OFF             4

 
typedef struct _GPIO_PIN {
    unsigned int port;
    unsigned int value;
    unsigned int bit;
} gpio_pin;

int IOInit();
int LEDOn();
int SpeakerON();
int SignleLEDOn();
int SixLEDOn();
int LEDOff();
int SetRedLED(int mValue);
int StartLEDQuickFlick();
int StopLEDQuickFlick();
int BatteryChargeOn();
int BatteryChargeOff();
char GetBtVal();
int SetBtOn(char mVal);
unsigned int GetPowerOff();
int ReadIOData(void *data, int size);
int GetResetBtValue();
int CloseIO();

#endif
