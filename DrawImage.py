import numpy as np
import cv2

class DrawImage():
	def __init__(self, uidSeq, ROILabelColor, coordSeq, imageSize, className):
		self.uidLen = len(uidSeq)
		self.coordSeq = coordSeq
		self.ROILabelColor = ROILabelColor
		self.DicomRTContourMap = np.zeros((self.uidLen, imageSize[0], imageSize[1], 3), dtype='uint8')
		self.DicomRTMap = np.zeros((self.uidLen, imageSize[0], imageSize[1], 3), dtype='uint8')
		self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
		self.className = className

	def __call__(self):
		self.drawContourColor()
		self.colorFillUp()
		return [self.DicomRTContourMap, self.DicomRTMap]
	
	def drawContourColor(self):
		itemsOfNumber = len(self.coordSeq)
		for i in range(itemsOfNumber):
			coordSeq = self.coordSeq[i]
			for j in range(len(coordSeq)):
				x, y, z = coordSeq[j]
				try:
					self.DicomRTContourMap[int(z),int(x),int(y)] = self.ROILabelColor
				except:
					print('self.ROILabelColor')
	
	def colorFillUp(self):
		itemsOfNumber = len(self.coordSeq)
		for i in range(self.uidLen):
			imgray = cv2.cvtColor(self.DicomRTContourMap[i], cv2.COLOR_BGR2GRAY)
			imgray = cv2.dilate(imgray,self.kernel,iterations = 1)
			ret,thresh = cv2.threshold(imgray,1,255,0)
			# img, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			_, cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			# cv2.drawContours(self.DicomRTMap[i], cnt, -1, [255, 255, 255], -1)




			
			if self.className == 'Liver':
				cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
			elif (self.className == 'Lung-L') or (self.className == 'Lung-R'):
				cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
			elif ((self.className == 'BODY')):
			 	cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
			elif ((self.className == 'Brain')):
			 	cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)    
			elif ((self.className == 'Brain Stem')):
			 	cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)                  
			elif (self.className == 'Kidney-L') or (self.className == 'Kidney-R'):
			 	cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
			elif ((self.className == 'Esophagus')):
				cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
			elif ((self.className == 'Stomach')):
				cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
			elif ((self.className == 'Heart')):
				cv2.drawContours(self.DicomRTMap[i], cnt, -1, self.ROILabelColor, -1)
            
                
