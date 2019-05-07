import cv2
import sys
import os.path
import numpy as np



def drawMatches(img1, kp1, img2, kp2, matches):

	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
	out[:rows1,:cols1] = np.dstack([img1])
	out[:rows2,cols1:] = np.dstack([img2])
	for mat in matches:
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

	return out


src_pts = dst_pts = []



def get_morpholy():
	global src_pts,dst_pts,img1
	#print(src_pts.shape,dst_pts.shape)
	if len(src_pts)>10:
		M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h,w = img1.shape[:2]
		
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		dst = cv2.perspectiveTransform(pts,M)

		perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
		result = cv2.warpPerspective(img2,perspectiveM,(w,h))#,borderMode=cv2.BORDER_CONSTANT,borderValue=[0,0,0,0])

		#_, _, _, _,result_crop = crop_image(result)
		min_y1,max_y2,min_x1,max_x2,_ = crop_image(result)
		print(min_y1,max_y2,min_x1,max_x2)

		cv2.imshow("img_src",img1)
		cv2.imshow("result",result)
		
		if max_y2>min_y1 and max_x2>min_x1:
			result_crop = result[min_y1:max_y2,min_x1:max_x2]
			img_crop = img1[min_y1:max_y2,min_x1:max_x2]
			loss = (100-img_crop.size/img1.size*100)
			print("loss: %.2f%%" % loss)
			

			
			cv2.imshow("img_crop",img_crop)
			cv2.imshow("result_crop",result_crop)
			if loss<90:
				crop_mini = cv2.resize(result_crop,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
				cv2.imshow("result_crop_mini",crop_mini)
			return img_crop, result_crop
		else:
			print("No success crop")

	else:
		print("No enought points")

	return [],[]

def click_event(event, x, y, flags, param):
	global src_pts, dst_pts, img1_width
	# grab references to the global variables
	
	#print(event,cv2.EVENT_RBUTTONUP)
	if event == cv2.EVENT_LBUTTONUP:
		coords = np.array([y,x],dtype=np.float32).reshape(-1,1,2)
		src_pts = np.concatenate((src_pts,coords))

	elif event == cv2.EVENT_RBUTTONUP:
		coords = np.array([y,x],dtype=np.float32).reshape(-1,1,2)
		dst_pts = np.concatenate((dst_pts,coords))
		
		get_morpholy()


##### CROP IMAGES

def crop_image(img):
	img_tmp = img.copy()
 
	hh,ww = img.shape[:2]
	ratio = ww/hh
 
	center_w, center_h = ww//2, hh//2
	start_width = ww//2
	 
	pto11_flag = True
	pto12_flag = True
	pto21_flag = True
	pto22_flag = True
	 
	pto11=pto12=pto21=pto22 = [0,0]
	min_y1=max_y2=min_x1=max_x2=0
	
	_,thresh = cv2.threshold(img_tmp,1,255,cv2.THRESH_BINARY)

	
	kernel = np.ones((5,5),np.uint8)
	'''mask = np.zeros((hh+2,ww+2),np.uint8)
	thresh_fill = thresh.copy()
	cv2.floodFill(thresh_fill,mask,(100,120),255)
	thresh_inv = cv2.bitwise_not(thresh_fill)
	thresh = thresh_fill | thresh_inv'''

	#cv2.imshow('thresh1', thresh)
	#thresh = cv2.erode(thresh,kernel,iterations = 2)
	#thresh = cv2.dilate(thresh,kernel,iterations = 2)
	#cv2.imshow('thresh2', thresh)

	jumps = 5
	 
	for i in range(100):
		width_tmp = start_width+jumps*i
		height = width_tmp/ratio*1

		pto11_tmp = [int(center_w-width_tmp//2),int(center_h-height//2)]
		pto12_tmp = [int(center_w+width_tmp//2),int(center_h-height//2)]
		pto21_tmp = [int(center_w-width_tmp//2),int(center_h+height//2)]
		pto22_tmp = [int(center_w+width_tmp//2),int(center_h+height//2)]

		if (pto11_tmp[0]<0 or pto11_tmp[1]<0): 
			pto11_flag = False
		elif thresh[pto11_tmp[1],pto11_tmp[0]]==0:
			pto11_flag = False
		if (pto12_tmp[0]>ww-jumps or pto12_tmp[1]<0):
			pto12_flag = False
		elif thresh[pto12_tmp[1],pto12_tmp[0]]==0:
			pto12_flag = False
		if (pto21_tmp[0]<0 or pto21_tmp[1]>hh-jumps): 
			pto21_flag = False
		elif thresh[pto21_tmp[1],pto21_tmp[0]]==0:
			pto21_flag = False
		if (pto22_tmp[0]>ww-jumps or pto22_tmp[1]>hh-jumps):
			pto22_flag = False
		elif thresh[pto22_tmp[1],pto22_tmp[0]]==0:
			pto22_flag = False
		 
		if pto11_flag: pto11 = pto11_tmp
		if pto12_flag: pto12 = pto12_tmp
		if pto21_flag: pto21 = pto21_tmp
		if pto22_flag: pto22 = pto22_tmp
		
		#print(pto11_flag,pto12_flag,pto21_flag,pto22_flag)
		if pto11_flag == pto12_flag == pto21_flag == pto22_flag == False: break
		
		 
		#cv2.rectangle(img, tuple(pto11), tuple(pto22),(255, 0, 0), 1)
		'''cv2.circle(img, (center_w,center_h), 2, (255, 255, 255, 2), 1)   
		cv2.circle(img, tuple(pto11), 2, (255, 255, 255, 2), 1)   
		cv2.circle(img, tuple(pto12), 2, (255, 255, 255, 2), 1)   
		cv2.circle(img, tuple(pto21), 2, (255, 255, 255, 2), 1)   
		cv2.circle(img, tuple(pto22), 2, (255, 255, 255, 2), 1)'''
		 
	'''if pto11[0]<pto21[0]:
		if img[pto21[1],pto11[0]] > 0:
			min_x1=pto21[0]
		else:
			min_x1=pto11[0]
	else:
		if img[pto11[1],pto21[0]] > 0:
			min_x1=pto21[0]
		else:
			min_x1=pto11[0]

	pto11[0] = min_x1
	pto21[0] = min_x1			
	 
	if pto11[1]<pto12[1]:
		if img[pto11[1],pto12[0]] > 0:
			min_y1=pto11[1]
		else:
			min_y1=pto12[1]
	else:
		if img[pto12[1],pto11[0]] > 0:
			min_y1=pto12[1]
		else:
			min_y1=pto11[1]
	 
	pto11[1] = min_y1
	pto12[1] = min_y1
	 
	 
	if pto12[0]>pto22[0]:
		if img[pto22[1],pto12[0]] > 0:
			max_x2=pto22[0]
		else:
			max_x2=pto12[0]
	else:
		if img[pto12[1],pto22[0]] > 0:
			max_x2=pto12[0]
		else:
			max_x2=pto22[0]
	
	pto12[0] = max_x2
	pto22[0] = max_x2

	if pto21[1]>pto22[1]:
		if img[pto21[1],pto22[0]] > 0:
			max_y2=pto21[1]
		else:
			max_y2=pto22[1]
	else:
		if img[pto22[1],pto21[0]] > 0:
			max_y2=pto22[1]
		else:
			max_y2=pto21[1]
		 
	pto21[1] = max_y2
	pto22[1] = max_y2'''

		#print(min_x1,min_y1,max_x2,max_y2,pto11,pto21,pto12)
		 
		#cv2.imshow('image2', img_tmp)
		#cv2.waitKey(100)
	
	pto11[1] = max(pto11[1],pto12[1])
	pto12[1] = max(pto11[1],pto12[1])

	pto21[1] = min(pto21[1],pto22[1])
	pto22[1] = min(pto21[1],pto22[1])

	pto11[0] = max(pto11[0],pto21[0])
	pto21[0] = max(pto11[0],pto21[0])

	pto12[0] = min(pto12[0],pto22[0])
	pto22[0] = min(pto12[0],pto22[0])

	min_y1 = pto11[1];max_y2 = pto22[1]
	min_x1 = pto11[0];max_x2 = pto22[0]

	img = img[min_y1:max_y2,min_x1:max_x2].copy()
	#cv2.imshow('crop', img2)
	 
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return min_y1,max_y2,min_x1,max_x2,img


# Initiate SIFT detector
def sift_surf_feature(img1,img2,mtype="SIFT",max_match=0.5,show=True):
	if mtype=="SIFT":
		method = cv2.xfeatures2d.SIFT_create()
	else:
		method = cv2.xfeatures2d.SURF_create()  #(400, 5, 5)

	# find the keypoints and descriptors with SIFT
	kp1, des1 = method.detectAndCompute(img1,None)
	kp2, des2 = method.detectAndCompute(img2,None)

	# BFMatcher with default params
	bf = cv2.BFMatcher(crossCheck=True) 
	#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	#bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key=lambda val: val.distance)
	#match_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=2)
	match_img = drawMatches(img1,kp1,img2,kp2,matches[:20])
	
	if show:
		cv2.imshow(mtype+' Matched Features', match_img)
		cv2.setMouseCallback(mtype+' Matched Features', click_event)
		#cv2.waitKey(0)

	index_params = dict(algorithm = 0, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)

	good = []
	for m,n in flann.knnMatch(des1,des2,k=2):
		#print(m.distance ,n.distance, 0.8*n.distance)
		if m.distance < max_match*n.distance:  #lower, less common points
			good.append(m)

	img1_pts = img2_pts = np.zeros((0,2),dtype=np.float32).reshape(-1,1,2)

	print(mtype+" pts:",len(good))

	if len(good)>8:
		img1_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
		img2_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
		#print("sift",len(src_pts))
		
		
		#print((w,h),result.shape)
		#print(np.int32(dst))
		#result2 = cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
		#cv2.imshow("result2"+name,result2)
		#print(pts)
		#xx,yy =pts[-1,0,0],pts[-1,0,1]
		#print("cords",xx,yy)
		#cv2.circle(result, (yy,xx), 3, (255,255,0), -1)
		
		#cv2.imwrite("result.png", result)
		#cv2.imshow("result"+name,result)
		#cv2.waitKey(0)

	return img1_pts,img2_pts


def orb_feature(img1,img2,max_match=0.10,show=True):
	orb = cv2.ORB_create(100)

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(des1,des2,None)
	matches.sort(key=lambda x:x.distance,reverse=False)
	numGoodMatches = int(len(matches)*max_match)  #lower, less points
	matches = matches[:numGoodMatches]
	match_img = cv2.drawMatches(img1,kp1,img2,kp2, matches, None)

	print("ORB pts:",len(matches))
	
	if show:
		cv2.imshow('ORB Matched Features', match_img)
		#cv2.waitKey(0)

	img1_pts = np.zeros((len(matches),2),dtype=np.float32).reshape(-1,1,2)
	img2_pts = np.zeros((len(matches),2),dtype=np.float32).reshape(-1,1,2)

	for i, match in enumerate(matches):
		img1_pts[i,:] = kp1[match.queryIdx].pt
		img2_pts[i,:] = kp2[match.trainIdx].pt

	

	#M, mask = cv2.findHomography(img1_pts,img2_pts, cv2. RANSAC)
	#h,w = img2.shape[:2]
	#img_orb = cv2.warpPerspective(img1,M,(w,h))
	#cv2.imshow("orb_result",img_orb)

	#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	#dst = cv2.perspectiveTransform(pts,M)
	#perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
	#result = cv2.warpPerspective(img2,perspectiveM,(w,h))
	#cv2.imshow("orb_result2",result)

	#if len(good)>10:
	return img1_pts,img2_pts


def compare(img1, img2):
	global src_pts,dst_pts
	#img2 = cv2.resize(img2,(640,480),interpolation=cv2.INTER_CUBIC)
	
	sift_img1_pts,sift_img2_pts = sift_surf_feature(img1,img2,"SIFT",True)
	surf_img1_pts,surf_img2_pts = sift_surf_feature(img1,img2,"SURF",True)
	orb_img1_pts,orb_img2_pts = orb_feature(img1,img2,True)
	
	src_pts = np.concatenate((sift_img1_pts,surf_img1_pts,orb_img1_pts))
	dst_pts = np.concatenate((sift_img2_pts,surf_img2_pts,orb_img2_pts))
	

	print("Total pts:",len(src_pts))


	#return src_pts,dst_pts

#def move_params(x):
#	global img1,img2
#	compare(img1,img2)
#	get_morpholy()


#cv2.namedWindow('Controls',cv2.WINDOW_NORMAL)
#cv2.createTrackbar("SIFT","Controls",0,100,move_params)
#cv2.setTrackbarPos("SIFT","Controls",40)




dataset_folder = "dataset2"

if os.path.exists(dataset_folder):
	files_domo = [dataset_folder+"/160_domo/"+img for img in os.listdir(dataset_folder+"/160_domo") if img.endswith(".jpg")]
	files_axis = [dataset_folder+"/320_axis/"+img for img in os.listdir(dataset_folder+"/320_axis") if img.endswith(".jpg")]
	#files_flir = [dataset_folder+"/640_flir/"+img for img in os.listdir(dataset_folder+"/640_flir") if img.endswith(".jpg")]

	for i in range(len(files_domo)):
		img1 = cv2.imread(files_axis[i],0)
		img2 = cv2.imread(files_domo[i],0)
		#img1 = cv2.resize(img1,(340,280),interpolation=cv2.INTER_CUBIC)
		#img2 = cv2.resize(img2,(340,280),interpolation=cv2.INTER_CUBIC)

		img1_heigth,img1_width = img1.shape[:2]
		#img1 = abs(255-img1)
		#img2 = abs(255-img2)


		compare(img1,img2)
		get_morpholy()
		
		#img1_heigth,img1_width = img1.shape[:2]
		#compare(img1,img2)
		#get_morpholy()


		key = cv2.waitKey(0)
		if key==27: exit()

else:
	exit("No folder found")




















exit()


#from __future__ import print_function

import cv2
import numpy as np
 

print(cv2.__version__)
############### Image Matching ###############


def match_images(img1, img2):
	"""Given two images, returns the matches"""
	sift = cv2.xfeatures2d.SIFT_create()
	detector = cv2.xfeatures2d.SURF_create(400, 5, 5)
	matcher = cv2.BFMatcher(cv2.NORM_L2)

	kp1, desc1 = detector.detectAndCompute(img1, None)
	kp2, desc2 = detector.detectAndCompute(img2, None)

	raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) 
	kp_pairs = filter_matches(kp1, kp2, raw_matches)
	return kp_pairs

def filter_matches(kp1, kp2, matches, ratio = 0.75):
	mkp1, mkp2 = [], []
	for m in matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			m = m[0]
			mkp1.append( kp1[m.queryIdx] )
			mkp2.append( kp2[m.trainIdx] )
	kp_pairs = zip(mkp1, mkp2)
	return kp_pairs

############### Match Diplaying ###############

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = numpy.zeros((max(h1, h2), w1+w2), numpy.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1+w2] = img2
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	if H is not None:
		corners = numpy.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
		corners = numpy.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
		cv2.polylines(vis, [corners], True, (255, 255, 255))

	if status is None:
		status = numpy.ones(len(kp_pairs), numpy.bool_)
	p1 = numpy.int32([kpp[0].pt for kpp in kp_pairs])
	p2 = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

	green = (0, 255, 0)
	red = (0, 0, 255)
	white = (255, 255, 255)
	kp_color = (51, 103, 236)
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			col = green
			cv2.circle(vis, (x1, y1), 2, col, -1)
			cv2.circle(vis, (x2, y2), 2, col, -1)
		else:
			col = red
			r = 2
			thickness = 3
			cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
			cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
			cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
			cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
	vis0 = vis.copy()
	for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
		if inlier:
			cv2.line(vis, (x1, y1), (x2, y2), green)
	cv2.imshow(win, vis)   
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def draw_matches(window_name, kp_pairs, img1, img2):
	"""Draws the matches for """
	mkp1, mkp2 = zip(*kp_pairs)

	p1 = np.float32([kp.pt for kp in mkp1])
	p2 = np.float32([kp.pt for kp in mkp2])

	if len(kp_pairs) >= 4:
		H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	else:
		H, status = None, None
	if len(p1):
		explore_match(window_name, img1, img2, kp_pairs, status, H)

############### Test ###############

img1 = cv2.imread('orig.jpg', 0)
img2 = cv2.imread('rota.jpg', 0)
cv2.imshow('REAL',img1)
cv2.imshow('Rotated',img2)
img_1= cv2.resize(img1,(500,600))
img_2= cv2.resize(img2,(500,600))
kp_pairs = match_images(img1, img2)
if kp_pairs:
	draw_matches('Matching Features', kp_pairs, img1, img2)	
else:
	print ("No matches found")

##  Crop the matched region from scene
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(img2,perspectiveM,(w,h))
cv2.imwrite("found.png", found)