import pydicom
import cv2
import os
import numpy as np
from pydicom.data import get_testdata_files
from RTtoImage import RTtoImage
from DrawImage import DrawImage


global Organ
global Lung
global Kidney  

Organ = ['Liver','Heart','Esophagus','Stomach']
#Organ = ['Liver']
#Organ = ['Esophagus']
#Organ = ['Heart']
#Organ = []
Lung = ['Lung-R','Lung-L','LUNG-L','LUNG-R']
Lung = []
Kidney = ['Kidney-R','Kidney-L']
#Kidney = []

class getArray(object):

	def createFile(fileName):
		if os.path.exists(fileName): # if the path already exists do nothing
			pass
		else:
			try:
				os.makedirs(fileName)       #創建資料夾
			except IOError:
				print(fileName + '<- This name causes an error')

	def CT_Array(self, CT_sum, WindowsLevel, WindowsWidth):

		#path = self.ds.BodyPartExamined         #Body Part Examined(dicom資料):身體部位檢查(CHEST:胸腔)
		#print('--CT data---: ' , path)

		PS0 = float(self.ds.PixelSpacing[0])    #Pixel Spacing:像素間距
		PS1 = float(self.ds.PixelSpacing[1])
		STK = float(self.ds.SliceThickness)     #Slice Thickness:切片厚度

		# Load dimensions based on number of rows columns and slices
		ConstPixelSpacing = (PS0, PS1, STK)
		ConstPixelDims = (int(self.ds.Rows), int(self.ds.Columns), len(CT_sum))

		# DicomArray = np.zeros(ConstPixelDims,dtype=self.ds.pixel_array.dtype)
		DicomArray = np.zeros(ConstPixelDims,dtype = np.float32)
		print('this dcm len: ', len(CT_sum))

		
		for filenameDCM in CT_sum:
			try:
				ds = pydicom.dcmread(filenameDCM, force=True)

			except:
				print(filenameDCM, '==== can\'t open this dcm file in CT_Array function')

	
    # HU = Pixel Value × Rescale Slope + Rescale Intercept
			HU = (np.float32)( ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept )
			DicomArray[:, :, CT_sum.index(filenameDCM)] = HU

			# print(self.ds.pixel_array.dtype)
			# cv2.imwrite( self.SavePath + '{:04}'.format(CT_sum.index(filenameDCM)) + '.tif', DicomArray[:, :, CT_sum.index(filenameDCM)])
			# print(CT_sum.index(filenameDCM))

    # HU值轉換成數位影像的灰階值域(0~255)
    # Window Level 與 Window Width，代表灰階範圍的中心值，以及有效灰階範圍的上下限   
		lower = (WindowsLevel - 0.5 * WindowsWidth)
		upper = (WindowsLevel + 0.5 * WindowsWidth)

		dFactor = 255.0/(np.float64)(upper - lower)

		index_upper = np.where(DicomArray>upper)
		index_lower = np.where(DicomArray<lower)
		index_inner = np.where((~(DicomArray<lower) & ~(DicomArray>upper)))
		DicomArray[index_upper] = 255.0
		DicomArray[index_lower] = 0.0
		DicomArray[index_inner] = (DicomArray[index_inner] - lower)*dFactor


		DicomArray = np.array(DicomArray, dtype=np.uint8())

		# eqHist_DicomArray = np.zeros(ConstPixelDims,dtype = np.uint8())
		# for filenameDCM in CT_sum:
		# 	gray = DicomArray[:, :, CT_sum.index(filenameDCM)]
		# 	eqHist  = cv2.equalizeHist(gray)
		# 	eqHist_DicomArray[:,:, CT_sum.index(filenameDCM)] = eqHist
		# eqHist_DicomArray = np.array(eqHist_DicomArray, dtype=np.float32)

			# cv2.imwrite( self.SavePath + '{:05}'.format(CT_sum.index(filenameDCM)) + '.bmp', eqHist_DicomArray[:,:, CT_sum.index(filenameDCM)])

		# for slice_ in range(np.size(DicomArray, 2)):
		# 	for i in range(np.size(DicomArray, 0)):
		# 		for j in range(np.size(DicomArray, 1)):		
		# 			if (DicomArray[i,j,slice_] < lower):
		# 				DicomArray[i,j,slice_] = 0;
		# 			elif (DicomArray[i,j,slice_] > upper):
		# 				DicomArray[i,j,slice_] = 255
		# 			else:
		# 				DicomArray[i,j,slice_] = ( DicomArray[i,j,slice_] - lower) * dFactor 	

		getArray.createFile(self.SavePath)
		path = self.SavePath +'DicomArray.npy'
		print('this array of max:', np.amax(DicomArray))
		print('this array of min:', np.amin(DicomArray))
		print('data type : ', DicomArray.dtype)
		np.save(path, DicomArray)


	def __init__(self, filePath, DCM_Folder_ID, SavePath):
		# input CT or RT data
		self.filePath = filePath
		self.DCM_Folder_ID = DCM_Folder_ID
		self.SavePath = SavePath

		if type(filePath) == list:
			self.isCT = True
			try:
				self.ds = pydicom.dcmread(self.filePath[0], force=True)  #讀取根據DICOM文件格式存儲的DICOM數據集。
			except:
				print('\n')
				print(self.filePath + 'this data can\'t open')
		else:
			self.isCT = False			
			try:
				self.ds = pydicom.dcmread(self.filePath, force=True)
			except:
				print('\n')
				print(self.filePath + 'this data can\'t open')
		print('\n')
		# print(self.ds.PixelSpacing)

	def __call__(self):

		if self.isCT == True:
			print('\n')
			print('===== into CT_Array function =====')
			print(self.ds.PatientID  , self.ds.RescaleSlope )
			print(self.ds.PatientID  , self.ds.RescaleIntercept )
			self.CT_Array(self.filePath, WindowsLevel=-100, WindowsWidth=400)
			print('===== outof CT_Array function =====')
			return 
			# os._exit(0)
		else:
			print('\n')
			print('===== into RTtoImage function =====')
			[uidSeq, ROILabel, ROILabelColor, coordSeqForImage, 
			imageSize, ContourSequenceIsNull, ROIObservationLabeliIsNull] = RTtoImage(self.ds, self.DCM_Folder_ID)()
			print('===== goout RTtoImage function =====')
		#creatFile
		for file_name in ROILabel:
			path_1 = self.SavePath + 'Original/' + file_name
			path_2 = self.SavePath + 'GroundTruth/' + file_name
			getArray.createFile(path_1)
			getArray.createFile(path_2)
		getArray.createFile(self.SavePath + 'GroundTruth/' + 'Lung')
		getArray.createFile(self.SavePath + 'Original/' + 'Lung')
		getArray.createFile(self.SavePath + 'GroundTruth/' + 'Kidney')
		getArray.createFile(self.SavePath + 'Original/' + 'Kidney')

		#Write image to file
		print('===== into DrawImage function=====')	
		list_Lung_GT = []
		list_Lung_OG = []
		list_Kidney_GT = []
		list_Kidney_OG = []        
		for i in range(len(ROILabel)):
			if (ROILabel[i] in Organ):
				original, groundtruth = DrawImage(uidSeq, ROILabelColor[i], coordSeqForImage[i], imageSize, ROILabel[i])()
				for j in range(len(uidSeq)):			
					cv2.imwrite(self.SavePath + 'Original/' + ROILabel[i] + '/' + uidSeq[j] + '.bmp', original[j])
					cv2.imwrite(self.SavePath + 'GroundTruth/' + ROILabel[i] + '/' + uidSeq[j] + '.bmp', groundtruth[j])
			elif (ROILabel[i] in Lung):
				original, groundtruth = DrawImage(uidSeq, ROILabelColor[i], coordSeqForImage[i], imageSize, ROILabel[i])()
				list_Lung_GT.append(groundtruth)
				list_Lung_OG.append(original)
				for j in range(len(uidSeq)):			
					cv2.imwrite(self.SavePath + 'Original/' + ROILabel[i]+ '/' + uidSeq[j] + '.bmp', original[j])
					cv2.imwrite(self.SavePath + 'GroundTruth/' + ROILabel[i] + '/' + uidSeq[j] + '.bmp', groundtruth[j])       
			elif (ROILabel[i] in Kidney):
				original, groundtruth = DrawImage(uidSeq, ROILabelColor[i], coordSeqForImage[i], imageSize, ROILabel[i])()
				list_Kidney_GT.append(groundtruth)
				list_Kidney_OG.append(original)
				for j in range(len(uidSeq)):			
					cv2.imwrite(self.SavePath + 'Original/' + ROILabel[i]+ '/' + uidSeq[j] + '.bmp', original[j])
					cv2.imwrite(self.SavePath + 'GroundTruth/' + ROILabel[i] + '/' + uidSeq[j] + '.bmp', groundtruth[j])                           
		# print("lung ",np.array(list_Lung_GT).shape)	


		# Lung-L&Lung-R合成一個Lung
		if (list_Lung_GT !=[]) and (list_Lung_OG!=[]):
			for j in range(len(uidSeq)):
				LungImage_GT = cv2.add(list_Lung_GT[0][j], list_Lung_GT[1][j])
				LungImage_OG = cv2.add(list_Lung_OG[0][j], list_Lung_OG[1][j])
		 	# GrayImage = cv2.cvtColor(LungImage, cv2.COLOR_BGR2GRAY)
				cv2.imwrite(self.SavePath + 'GroundTruth/' + 'Lung' + '/' + uidSeq[j] + '.bmp', LungImage_GT)
				cv2.imwrite(self.SavePath + 'Original/' + 'Lung' + '/' + uidSeq[j] + '.bmp', LungImage_OG)

		# Kidney-L & Kidney-R 合成一個Kidney     
		if (list_Kidney_GT !=[]) and (list_Kidney_OG!=[]):
			for j in range(len(uidSeq)):
				KidneyImage_GT = cv2.add(list_Kidney_GT[0][j], list_Kidney_GT[1][j])
				KidneyImage_OG = cv2.add(list_Kidney_OG[0][j], list_Kidney_OG[1][j])
		 	# GrayImage = cv2.cvtColor(LungImage, cv2.COLOR_BGR2GRAY)
				cv2.imwrite(self.SavePath + 'GroundTruth/' + 'Kidney' + '/' + uidSeq[j] + '.bmp', KidneyImage_GT)
				cv2.imwrite(self.SavePath + 'Original/' + 'Kidney' + '/' + uidSeq[j] + '.bmp', KidneyImage_OG)   
           
		print('===== goout DrawImage function =====')	

		#All label color
		color_text_file = open(self.SavePath + 'Readme.txt', 'w')
		for i in range(len(ROILabel)):
			color_text_file.write(ROILabel[i] + ':' + str(ROILabelColor[i]) + '\n')
		color_text_file.write('==========================================' + '\n')
		for i in ContourSequenceIsNull:
			color_text_file.write(i + ' ContourSequence is [] ' + '\n')
		color_text_file.write('==========================================' + '\n')
		for i in ROIObservationLabeliIsNull:
			color_text_file.write(str(i) +  " ROI Observation Label is [] " + '\n')




		
