# AVE_learning-a-representation-for-navigation



1.cut the viseo into shorter pieces:
ffmpeg -ss 0:0:37 -i input.mp4 -t 0:3:20  -c copy out.mp4 

2. extract 1000 images from the cut video.
  ffmpeg -i out.mp4 -vf fps=1000/3.33*60 %d.png
