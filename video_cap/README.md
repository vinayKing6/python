Preview of video capture!

What you need:
		
		1.python3 
				sudo apt install python3

		2.opencv
				sudo apt install python3-opencv

How to run:
		
		import videoCap in your python files

		python3 filename.py

What's the function:
		
		to output images from a video

		_init_(self,path)
				path: path of processing video

		read_info(self)
				print information of processing video including fps ,total frames,time duration,width height

		image_output(self,frame_flage)
				frame_flage: to output images per specified frames

		video_close
				close invoking video
		
