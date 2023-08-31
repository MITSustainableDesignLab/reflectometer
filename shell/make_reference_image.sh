#!/usr/bin/env bash

# This script produces reference images for use with reflectometer.py
# Author: Nathaniel Jones
#
# Usage:
# make_reference_image.sh [-p|-a] [-s|-t] [-f file] [-x resolution] [-vh angle]
#
# -p : Generate reference images for Radiance plastic model (default)
# -a : Generate reference images for Ashikhmin-Shirly model
# -s : Generate reference image for horizontal slot (default)
# -t : Generate reference image for vertical slot
# -f : Specify the Radiance scene to use (defaults to blackbox.rad)
# -x : Specify the image resolution (defaults to 512)
# -vh : Specify the horizontal angle of the image in degrees (defaults to 85.431961)

if grep -q Microsoft /proc/version; then
	# Set aliases to Windows programs
	shopt -s expand_aliases
	alias hdrgen='/mnt/c/Program_Foils/hdrgen/bin/hdrgen' # This should be the path to hdrgen on your computer or cluster
	alias pcomb=pcomb.exe
	alias pcompos=pcompos.exe
	alias pfilt=pfilt.exe
	alias falsecolor=falsecolor.exe
	alias wxfalsecolor=wxfalsecolor_v0.52.exe # Make sure to download http://tbleicher.github.io/wxfalsecolor/ if using this program
	alias ra_tiff=ra_tiff.exe
	alias ra_bmp=ra_bmp.exe
	alias getinfo=getinfo.exe
	alias pvalue=pvalue.exe
	alias total=total.exe
	alias rcalc=rcalc.exe
	alias oconv=oconv.exe
	alias rpict=rpict.exe
fi

export RAYPATH=.:$RAYPATH

# Default settings
ashik=false
transverse=false
rad=blackbox.rad
res=512
vh=85.431961
decades=5
palette=tbo

# Enclosure dimensions
slot_light=128.2 # mm left of center
light_height=148.6 # mm
camera_height=160.4774288 # mm

opts="-aa 0 -ab 0 -ad 1000 -lw 0.00001 -lr 10"

# Read command line arguments
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
		-a|--ashik)
		ashik=true
		shift # past argument
		;;
		-p|--plastic)
		ashik=false
		shift # past argument
		;;
		-s|--scanline)
		transverse=false
		shift # past argument
		;;
		-t|--transverse)
		transverse=true
		shift # past argument
		;;
		-f|--file)
		rad="$2"
		shift # past argument
		shift # past value
		;;
		-x|--resolution)
		res="$2"
		shift # past argument
		shift # past value
		;;
		-vh|--angle)
		vh="$2"
		shift # past argument
		shift # past value
		;;
		-n|--decades)
		decades="$2"
		shift # past argument
		shift # past value
		;;
		-p|--palette)
		palette="$2"
		shift # past argument
		shift # past value
		;;
		-o|--options)
		opts=$(echo "$2" | sed 's/,/ /g')
		shift # past argument
		shift # past value
		;;
		*)    # unknown option
		echo "Unknown arg $1" >&2
		shift # past argument
		;;
	esac
done

vv=$(bc <<< "scale=6; $vh/$res")
#echo -vh $vh -vv $vv

if [ "$ashik" = true ]; then
	imgD=ashik
else
	imgD=plastic
fi

if [ "$transverse" = true ]; then
	pi=$(bc -l <<< "scale=10; 4*a(1)")
	slot_highlight=$(bc <<< "scale=6; $slot_light*$camera_height/($camera_height+$light_height)") # mm
	vv_rad=$(bc <<< "scale=6; $vv*$pi/180") # mm
	image_height=$(bc -l <<< "scale=6; 2*$camera_height*s($vv_rad/2)/c($vv_rad/2)") # mm
	vl=$(bc <<< "scale=6; -$slot_highlight/$image_height") # mm
	# echo -vl $vl

	imgD="${imgD}_vertical"
	view="-vp 0 0 $camera_height -vd 0 0 -1 -vu 1 0 0 -vtv -vh $vh -vv $vv -vs 0 -vl $vl -bv"
else
	imgD="${imgD}_horizontal"
	view="-vp 0 0 $camera_height -vd 0 0 -1 -vu 0 1 0 -vtv -vh $vh -vv $vv -bv"
fi

filelist=

mkdir -p $imgD

hdr=$imgD/diffuse.hdr
if [ ! -s $hdr ]; then
	# Define diffuse material
	if [ "$ashik" = true ]; then
		model="void ashik2 sample 4 1 0 0 . 0 8 1 1 1 0 0 0 0 0 !xform $rad"
	else
		model="void plastic sample 0 0 5 1 1 1 0 0 !xform $rad"
	fi

	# Make diffuse image
	echo $model
	echo $model | oconv - | rpict $view $opts -x $res -y 1 > $hdr
fi
filelist="$filelist $hdr"

r_min=0
r_max=500
r_inc=1

as_max=10000

for r in $(seq $r_min $r_inc $r_max); do
	roughness=$(printf "%03d" $r)
	hdr=$imgD/rough$roughness.hdr

	if [ ! -s $hdr ]; then
		# Define specular material
		if [ "$ashik" = true ]; then
			as_rough=$(bc -l <<< "scale=6; $as_max*e(-$r*l($as_max)/$r_max)")
			model="void ashik2 sample 4 1 0 0 . 0 8 0 0 0 1 1 1 $as_rough $as_rough !xform $rad"
		else
			model="void plastic sample 0 0 5 0 0 0 1 0.$roughness !xform $rad"
		fi

		# Make specular image
		echo -e "\e[1A\e[K$model"
		echo $model | oconv - | rpict $view $opts -x $res -y 1 > $hdr
	fi
	filelist="$filelist $hdr"
done

# Combine scanlines
output=$imgD.hdr
pcompos -a 1 $filelist > $output

# Create falsecolor image
outtif=$imgD.tif
falsecolor -ip $output -n $decades -log $decades -m 1 -s 1 -pal $palette | ra_tiff - $outtif

echo -e '\e[1A\e[KDone'
