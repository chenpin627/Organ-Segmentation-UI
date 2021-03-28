import tempfile
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset

#Multip_Organ = ['Lung','Liver','Stomach','Esophagus','Heart','Kidney']



class ImagetoRT():
	def __init__(self, DataBase, items,  DICOMInformation, DICOM_RT,AILabel):
		print('init RT Info.')
		self.DataBase = DataBase
		self.items = items
		self.DICOMInformation = DICOMInformation
		self.numberOfImage = len(DICOMInformation)
		self.itemLen = len(items)
		self.DataBaseLen = len(DataBase)
		self.DICOM_RT = DICOM_RT
		self.ROINumberStart = len(items)
		self.AILabel = AILabel

	def __call__(self):
		self.setStructureSetROISequence()
		self.setROIContourSequence()
		self.setRTROIObservationsSequence()
		# self.setReferencedFrameOfReferenceSequence()
		print('set done.')
		return self.DICOM_RT


	def setROIContourSequence(self):
		print('set ROI Contour Sequence.' )
		ROINumber = 500
		for i in range(self.DataBaseLen):
			MyROIContourSequence = pydicom.Dataset()
			ROINumber = ROINumber + 1
			# MyROIContourSequence.ROINumber = ROINumber
			MyROIContourSequence.ReferencedROINumber = ROINumber
			organ = str(self.AILabel[i])
			ContourSequence = self.getContourSequence(str(self.AILabel[i]))
			#print(ContourSequence)
			MyROIContourSequence.ContourSequence = ContourSequence
			color = self.DisplayColor(organ)
			MyROIContourSequence.ROIDisplayColor = color
			self.DICOM_RT.ROIContourSequence.append(MyROIContourSequence)

	def DisplayColor(self ,organ):
		if organ == 'AILung' :
			color = [255,174,53]
		elif organ == 'AILiver' :   
			color = [243,201,73]
		elif organ == 'AIStomach' : 
			color = [160,0,160]
		elif organ == 'AIEsophagus' :   
			color = [170,255,170]
		elif organ == 'AIHeart' :  
			color = [176,255,255]
		elif organ == 'AIKidney' :  
			color = [0,102,0]    
		return color


	def getDataCoordination(self, name):
		#l = len(self.DataBase)
		for i in range(self.itemLen):
			if(self.DataBase[i][0] == name):
				data = self.DataBase[i][1]
				break
		return data

	def getSOPInstanceUID(self, z_axis):
		SOPInstanceUID = -1
		for i in range(self.numberOfImage):
			if(self.DICOMInformation[i].SliceLocation == z_axis):
				SOPInstanceUID = self.DICOMInformation[i].SOPInstanceUID
				break
		return SOPInstanceUID

	def getContourSequence(self, itemName):
		print('get Contour Sequence.' )
		contourData = self.getDataCoordination(itemName)
		AllContourSequence = []
		# slice
		for numberOfContour in range(len(contourData)-1, -1, -1):
			# print(numberOfContour)
			data = contourData[numberOfContour]
			# contour pixel point
			for i in range(len(data)):
				pixelData = np.array(data[i]).reshape(-1)
				z_axis = pixelData[2]          
				contourImageSequence = pydicom.Dataset()
				contourImageSequence.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
				contourImageSequence.ReferencedSOPInstanceUID = self.getSOPInstanceUID(z_axis)      
				contourSequence = pydicom.Dataset()
				contourSequence.ContourImageSequence = [contourImageSequence]
				contourSequence.ContourGeometricType = "CLOSED_PLANAR"
				contourSequence.NumberOfContourPoints = len(data[i])
				contourSequence.ContourData = list(pixelData)
				AllContourSequence.append(contourSequence)
		return AllContourSequence

	def setRTROIObservationsSequence(self):
		print('set RT ROI Observations Sequence.' )
		ROINumber = 500
		for i in range(self.DataBaseLen):
			ROINumber = ROINumber + 1
			RTROIObservations = Dataset()
			RTROIObservations.ObservationNumber = 0
			RTROIObservations.ROIInterpreter = ""
			RTROIObservations.RTROIInterpretedType = "ORGAN"
			RTROIObservations.ReferencedROINumber = ROINumber
			RTROIObservations.ROIObservationLabel =str(self.AILabel[i])
			self.DICOM_RT.RTROIObservationsSequence.append(RTROIObservations)

	def setStructureSetROISequence(self):
		print('set Structure Set ROI Sequence.')
		ROINumber = 500
		items = self.items
		for i in range(self.DataBaseLen):
			ROINumber = ROINumber + 1
			structureSetROI = Dataset()
			structureSetROI.ROIGenerationAlgorithm = 'MANUAL'
			structureSetROI.ROIName = str(self.AILabel[i])
			#print(structureSetROI.ROIName)

			structureSetROI.ROINumber = ROINumber
			structureSetROI.ReferencedFrameOfReferenceUID = self.DICOMInformation[0].FrameOfReferenceUID
			self.DICOM_RT.StructureSetROISequence.append(structureSetROI)	

	# def setReferencedFrameOfReferenceSequence(self):
	# 	print('set Referenced Frame Of Reference Sequence.')
	# 	ReferencedFrameOfReference = Dataset()
	# 	ReferencedFrameOfReference.FrameOfReferenceUID = self.DICOMInformation[0].FrameOfReferenceUID
	# 	rtReferencedStudySequence = self.getRTReferencedStudySequence()
	# 	ReferencedFrameOfReference.RTReferencedStudySequence = [rtReferencedStudySequence]
	# 	self.ds.ReferencedFrameOfReferenceSequence.append(ReferencedFrameOfReference)
		
	# def getRTReferencedStudySequence(self):
	# 	RTReferencedStudy = Dataset()
	# 	RTReferencedSeries = Dataset()
	# 	contourImageSequence = self.getContourImageSequence()
	# 	RTReferencedSeries.ContourImageSequence = contourImageSequence
	# 	RTReferencedSeries.SeriesInstanceUID = "1.3.12.2.1107.5.1.4.29309.30000015101602373771800000464"
	# 	RTReferencedStudy.RTReferencedSeriesSequence = [RTReferencedSeries]
	# 	RTReferencedStudy.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.2"
	# 	RTReferencedStudy.ReferencedSOPInstanceUID = "1.2.410.200010.886.1140030012.68041014924942071"
	# 	return RTReferencedStudy
		
	# def getContourImageSequence(self):
	# 	contourImageSequence = []
	# 	for i in range(self.numberOfImage):
	# 		contourImage = Dataset()
	# 		contourImage.ReferencedSOPClassUID = self.DICOMInformation[i].file_meta.MediaStorageSOPClassUID
	# 		contourImage.ReferencedSOPInstanceUID = self.DICOMInformation[i].file_meta.MediaStorageSOPInstanceUID
	# 		contourImageSequence.append(contourImage)
	# 	return contourImageSequence