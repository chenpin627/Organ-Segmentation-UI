import numpy as np
import pydicom
import os
import re
import cv2
from natsort import natsorted


class SegmentiontoImageData():
	def __init__(self, DICOM_File, predict_File,itemColorList_txt):
		self.pixelSpacing = (1,1)
		self.initial_position = (1,1,1)
		self.DICOM_File_PATH = DICOM_File

		self.predict_File_PATH = predict_File
		self.itemColorList_txt = itemColorList_txt
		self.pixelDatabase = []
		self.items = []
		self.itemsLen = 0
		self.ROIColor = []
		self.DICOMInformation = []
		self.spacingDatabase = []
		self.contourSequence = []
		self.imageList = []
		self.numberOfImage = 0
		self.itemsdir=[]
		
	def __call__(self):
		print('get Contour Sequence from Images.', end='')  #從影像獲取輪廓
		self.getDICOMinformation()
		self.getItemColor()
		self.getImageData()
		self.pixeltoSpacing()
		print('....')
		return [self.spacingDatabase, self.items,self.ROIColor,self.DICOMInformation,self.itemsdir]
	
	def getSliceLocation(self, filename):
		l = len(self.DICOMInformation)
		number = -1
		for i in range(l):
			if(self.DICOMInformation[i].SOPInstanceUID == filename):
				number = self.DICOMInformation[i].SliceLocation
				
				break
		return number
		
	def getDICOMinformation(self):
		DICOMdir = os.listdir(self.DICOM_File_PATH)
		DICOMdir = natsorted(DICOMdir,reverse=False)			

		for DICOMName in DICOMdir:
			filename = self.DICOM_File_PATH + DICOMName

			dataset = pydicom.dcmread(filename)
			if(dataset.Modality == "CT"):
				self.DICOMInformation.append(dataset)
		self.pixelSpacing = self.DICOMInformation[0].PixelSpacing
		self.initial_position = self.DICOMInformation[0].ImagePositionPatient

	# def getItemColor(self):
	# 	with open(self.itemColorList_txt) as f:
	# 		content = f.read()
	# 	x = re.split(",|:|\[|\]|\*|\n|'",str(content))
	# 	for i in range(0, len(x)-1, 6):
	# 		item = x[i]
	# 		R = int(x[i+2].split('\"')[1])
	# 		G = int(x[i+3].split('\"')[1])
	# 		B = int(x[i+4].split('\"')[1])
	# 		self.items.append(item)
	# 		self.ROIColor.append([item, [B, G, R]])

	def getItemColor(self):
		with open(self.itemColorList_txt) as f:
			content = f.read()

		x = re.split(",|:|\[|\]|\*|\n|'",str(content))
		separation_count = x.count("==========================================")

		if separation_count :
			separation_index = x.index("==========================================")
			x = x[:separation_index]

		for i in range(0, len(x)-1, 6):
			item = x[i]
			R = int(x[i+2].split('\"')[1])
			G = int(x[i+3].split('\"')[1])
			B = int(x[i+4].split('\"')[1])
			self.items.append(item)
			self.ROIColor.append([item, [B, G, R]])

			
	def getImageData(self):
		self.itemsdir = os.listdir(self.predict_File_PATH)
		self.itemsdir = natsorted(self.itemsdir,reverse=False)	

		self.itemsLen = len(self.itemsdir)
		for item in self.itemsdir:
			path = self.predict_File_PATH + item
			imagesdir = os.listdir(path)
			imagesdir = natsorted(imagesdir,reverse=False)
			#print(len(imagesdir))
			self.numberOfImage = len(imagesdir)
			pixelData = []
			z_axis = -1
			for imageName in imagesdir:
				name = imageName[0:-4]
				# print(name)
				image = cv2.imread(path + "/" + imageName)
				imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				ret,thresh = cv2.threshold(imgray,127,255,0)
				#map = np.zeros((512, 512, 3))
				img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
				z_axis = z_axis + 1
				if(contours == []):
					continue
				temp = []
				for cnt in contours:
					cnt = list(np.array(cnt).reshape(-1))
					temp.append(cnt)
				pixelData.append([name, temp])
				#for cnt in contours:
					#pixelData.append(cnt)
					#cv2.drawContours(map, cnt, -1, (0, 255, 0), -1)
					#cv2.imshow("test",map)
					#cv2.waitKey(30)
			#print(item)
			#print(len(pixelData))
			self.pixelDatabase.append([item ,pixelData])


			
	def pixeltoSpacing(self):

		# organs
		for itemNumber in range(self.itemsLen):
			item, itemData = self.pixelDatabase[itemNumber]
			# print(itemData)
			database = []
			# organ's images
			for imageNumber in range(len(itemData)):
				contours = []
				name, contoursData = itemData[imageNumber]

				z_axis = self.getSliceLocation(name)
				# image's contour
				for numberOfContour in range(len(contoursData)):
					coordinate = []
					pixelData = contoursData[numberOfContour]
					# contour to real world coordinate
					for i in range(0, len(pixelData), 2):
						x = pixelData[i]
						y = pixelData[i+1]
						xx = x * self.pixelSpacing[0] + self.initial_position[0] 
						yy = y * self.pixelSpacing[1] + self.initial_position[1] 
						zz = z_axis
						coordinate.append([xx, yy, zz])
						# print(coordinate)
					contours.append(coordinate)
				database.append(contours)
			self.spacingDatabase.append([item, database])
		# print(self.spacingDatabase)