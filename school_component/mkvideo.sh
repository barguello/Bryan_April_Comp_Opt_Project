#!/bin/bash

# This is a script to create a video from series of JPEG images
# Call it in a folder full of JPEGs that you want to turn into a video.
# Written on 2013-01-08 by Philipp Klaus <philipp.l.klaus →AT→ web.de>.
# Check <https://gist.github.com/4572552> for newer versions.

# Resources
# * http://www.itforeveryone.co.uk/image-to-video.html
# * http://spielwiese.la-evento.com/hokuspokus/index.html
# * http://ffmpeg.org/trac/ffmpeg/wiki/Create%20a%20video%20slideshow%20from%20images
# * http://wiki.ubuntuusers.de/FFmpeg

set -x

FRAMERATE=4
RESOLUTION=1920x1080

# Rename the images into a sequence
# http://www.ralree.com/2008/08/06/renaming-files-sequentially-with-bash/
EII=1
# If sorting according to the file date, copy them using   cp -a ../*.JPG ./
for i in $(ls -tr *.jpg); do
  ls $i
  NEWNAME=plot_`printf "%05d" $EII`.jpg
  #echo Renaming $i to $NEWNAME
  mv $i $NEWNAME
  EII=`expr $EII + 1`
done

# Resize the images (replaces the files)
mogrify -resize $RESOLUTION *.JPG

# Now create the video using ffmpeg
cat *.JPG | ffmpeg -f image2pipe -r $FRAMERATE -vcodec mjpeg -i - -vcodec libx264 out_$FRAMERATE.mp4
#ffmpeg -f image2 -r $RATE -i IMG_%05d.JPG movie_$RATE.mp4
