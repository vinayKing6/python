import cv2 as cv,time

class video_capture:
    
    def __init__(self,path):
        self.path=path
        self.video_cap=cv.VideoCapture(self.path)
        self.frames_num=self.video_cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.video_fps=self.video_cap.get(cv.CAP_PROP_FPS)
        self.video_times=int(self.frames_num)/int(self.video_fps)
        self.frame_width=self.video_cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.frame_height=self.video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    def read_info(self):
        print(f"Video Info:\n video name: {self.path}\n fps: {self.video_fps}\n total frames: {self.frames_num}\n video time: {self.video_times}\n frame width: {self.frame_width} frame_height: {self.frame_height}" )

    def image_output(self,frame_flage):
        start=time.time()

        for i in range(int(self.frames_num)):
            ret=self.video_cap.grab()
            if not ret:
                print('Error!')
                break
            if i % int(frame_flage)==0:
                ret,frame=self.video_cap.retrieve()
                if ret:
                    print(i)
                    cv.imwrite('./images/image{}.jpg'.format(int(i/int(frame_flage))),frame)
            cv.waitKey(1)

        end=time.time()
        print(f"Running time: {end-start} secs\n")

    def video_close(self):
        self.video_cap.release()
