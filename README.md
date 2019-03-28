# [SimCam](https://simcam.ai/) AI Camera.

The first on-device AI Security Camera for smart home.

![camera pic](img/simcam.jpg "SimCam AI Camera")

<div style="text-align: justify">
The SimCam uses AI for facial recognition, pet monitoring, and more via location training. The SimCam has a 5 megapixel image sensor with night vision for still images and 1080 HD videos. The IP65 waterproof rated indoor/outdoor, camera can rotate 360 degrees while tracking objects.
</div>
With the open SDK of SimCam, you can customize the settings to meet your needs.

The work flow of SimCam in developer mode is shown in the following figure.

![work flow](img/simcam3.jpg "Work Flow of Camera")

<div style="text-align: justify">
As shown in the figure, SimCam consists of two embedded boards, one is HI3516 and another is Movidius. HI3516 is used for video capture and video play. Movidius is used for deep learning computing. Two embedded boards communicate by SPI.
</div>
There are some documents for developers inside [docs](./docs/) folder including :

   * [Quick start guide of SimCam in developer mode](docs/Quick_Start_Guide.pdf);
   * [Toolchain installation and usage guide](docs/Guide_of_Tool_Chain_Installation_and_usage.pdf);
   * [SimCam API guide](docs/Guide_of_SimCam_SDK_APIs.pdf);
   * [How to train object detection model](docs/How_To_Train_Model.pdf)


Developers can train their own object detection models using Caffe deep learning framework and  neural network architecture provided by SimCam team.

Instruction can be found in [this document](./docs/How_To_Train_Model.pdf).
However, SIMACAM team has provided several robust detection models, such as [baby climb](examples/models/babyclimb) detection model, [gesture](examples/models/gesture) detection model, [person car face](examples/models/person_car_face) detection model, [pet magic](examples/models/pet_magic) detection models.  Here is some interesting results of detection for some models:
<br>
Pet magic detection model:

![pet magic](https://github.com/RamatovInomjon/mygifs/blob/master/petmagicgif.gif "Pet magic test")

<br>
Baby climb detection model:

![baby climb](https://github.com/RamatovInomjon/mygifs/blob/master/babyclimb.gif "baby climb test")

### Support
If you need any help, please post us an issue on [Github Issues](https://github.com/simshineaicamera/SIMCAM_SDK/issues).  You are welcome to contact us for your suggestions!
