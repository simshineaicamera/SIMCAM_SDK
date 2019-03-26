"""
Created on Thu Nov 15 09:38:32 2018

@author: created by baijun, edited by Inomjon
"""
import os
import argparse,random
import cv2

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i', dest='input',help='video input path', type=str, default='videos/')
    parser.add_argument('--output','-o',dest='output',help='output path', type=str, default='JPEGImages/')
    parser.add_argument('--num','-n',dest='numFramePerSecond',type=int,default=1,help='num frame to get per second')
    return parser.parse_args()

def main(args):
    list_video=os.listdir(args.input)
    if not os.path.exists(args.output):
        os.system('mkdir -p %s'%args.output)
    for video_name in list_video:
        video_path=os.path.join(args.input,video_name)
        cap = cv2.VideoCapture(video_path)

        FPS=int(cap.get(cv2.CAP_PROP_FPS))
        
        random_index = random.sample(range(FPS), FPS)
        indexFrame=random_index[:args.numFramePerSecond]
        
        ret=True
        count=0
        numOfImage=0
        while ret:
            ret,frame = cap.read()
            if ret == False:
                break
            count+=1
            if count==FPS:
                count=0
            if count in indexFrame:
                numOfImage+=1
                video_name=video_name.split('.')[0]
                image_name=video_name+'_%08d'%numOfImage+'.jpg'
                image_path=os.path.join(args.output,image_name)
                cv2.imwrite(image_path,frame)
if __name__ == '__main__':
    args = parse_args()
    main(args)

