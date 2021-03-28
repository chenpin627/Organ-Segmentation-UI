import pydicom
import os
from natsort import natsorted
class RTtoImage():
	def __init__(self, ds, DICOM_File):
		self.uidSeq = []
		self.coordSeq = []
		self.coordSeqForImage = []
		self.all_coordSeqForImage = []
		self.ROILabel = []
		self.ROILabelColor = []
		self.DICOMInformation = []
		self.ds = ds
		self.DICOM_File = DICOM_File # RT plan File
		self.pixelSpacing = (1,1,1)
		self.initial_position = (1,1,1)
		self.imageSize = (0,0)
		self.item = len(ds.RTROIObservationsSequence) #num of organ that had labelled
		self.item_number = len(ds.RTROIObservationsSequence)
		self.ContourSequenceIsNull = []
		self.ROIObservationLabeliIsNull = []


	def __call__(self):
		self.getDICOMinformation()
		self.get_UIDSequence()
		self.get_ROILabelColor()
		self.get_ROILabel()
		for i in range(self.item):
			self.get_CoordinateSequence(i)
			self.spacingtoPixel(self.pixelSpacing, self.initial_position)
			self.all_coordSeqForImage.append(self.coordSeqForImage)
			self.coordSeq = []
			self.coordSeqForImage = []
		return [self.uidSeq, self.ROILabel, self.ROILabelColor, self.all_coordSeqForImage,
			    self.imageSize, self.ContourSequenceIsNull, self.ROIObservationLabeliIsNull
			    ]
		
		# if "SliceLocation" without given and we assumed to be empty
	def getDICOMinformation(self):
		DICOMdir = os.listdir(self.DICOM_File)
		DICOMdir = natsorted(DICOMdir, reverse=False)
		for DICOMName in DICOMdir:
			filename = self.DICOM_File + DICOMName
			try:
				dataset = pydicom.dcmread(filename, force=True)
			except:	
				print(filename + '  can\'t open')

			if(dataset.Modality == "CT"):
				self.DICOMInformation.append(dataset)
		pixelSpacing = self.DICOMInformation[0].PixelSpacing

		if(self.DICOMInformation[0].dir("SliceLocation") !=[]):
			sliceLocation_1 = self.DICOMInformation[0].SliceLocation
			sliceLocation_2 = self.DICOMInformation[1].SliceLocation
			#z_level = self.DICOMInformation[0].SliceThickness
			z_level = abs(sliceLocation_1 - sliceLocation_2)

		else:
			try:
				sliceLocation_1 = self.DICOMInformation[0].SliceLocation
				sliceLocation_2 = self.DICOMInformation[1].SliceLocation
				z_level = self.DICOMInformation[0].SliceThickness
			except Exception:
					z_level = self.DICOMInformation[0].ImagePositionPatient[2]
					print('patien position')
			
  
		#print(pixelSpacing[2])

		self.pixelSpacing = (pixelSpacing[0], pixelSpacing[1], z_level)
		self.initial_position = self.DICOMInformation[0].ImagePositionPatient
		self.imageSize = (self.DICOMInformation[0].Rows, self.DICOMInformation[0].Columns)
		
	def get_ROILabelColor(self):
		for i in range(self.item):
			labelColor = self.ds.ROIContourSequence[i].ROIDisplayColor
			color = [labelColor[2], labelColor[1], labelColor[0]]
			self.ROILabelColor.append(color)
			
	def get_ROILabel(self):
		for i in range(self.item):
			try:
				if(self.ds.RTROIObservationsSequence[i].RTROIInterpretedType == 'ORGAN'):
					label = self.ds.RTROIObservationsSequence[i].ROIObservationLabel
					self.ROILabel.append(label)
				else:
					label =self.ds.RTROIObservationsSequence[i].RTROIInterpretedType
					self.ROILabel.append(label)
			except:
				print(" - - -NO ROIObservationLabel - - -")
				print(self.ds.RTROIObservationsSequence[i] )
				self.ROIObservationLabeliIsNull.append(str(self.ds.RTROIObservationsSequence[i]))
			# self.ROILabel.append(label)
	
	#ReferencedFrameOfReferenceSequence => list all CT_image ID 
	def get_UIDSequence(self):
		frame_of_ref = self.ds.ReferencedFrameOfReferenceSequence[0]
		study = frame_of_ref.RTReferencedStudySequence[0]
		imageSequence = study.RTReferencedSeriesSequence[0]
		length = len(imageSequence.ContourImageSequence[:])
		for i in range(length):
			uid = imageSequence.ContourImageSequence[i].ReferencedSOPInstanceUID
			self.uidSeq.append(uid)
			
	def get_CoordinateSequence(self, item_number):
		ds = self.ds
		if(ds.ROIContourSequence[item_number].dir("ContourSequence") !=[]):
			contourOfNumber = len(ds.ROIContourSequence[item_number].ContourSequence)
		else:
			Label = ds.RTROIObservationsSequence[item_number].ROIObservationLabel
			print(Label + ' ContourSequence is []')
			self.ContourSequenceIsNull.append(Label)
			contourOfNumber = 0
			self.coordSeq.append([])
			
		for i in range(contourOfNumber-1, -1, -1):
			coordinate = []
			contour = ds.ROIContourSequence[item_number].ContourSequence[i].ContourData #ContourSequence sort from the large number
			number_of_points = ds.ROIContourSequence[item_number].ContourSequence[i].NumberOfContourPoints
			for j in range(0, number_of_points*3, 3):
				x = contour[j]
				y = contour[j+1]
				z = contour[j+2]
				coordinate.append([x, y, z])
			self.coordSeq.append(coordinate)
	
	def spacingtoPixel(self, pixelSpacing, initial_position):
		for i in range(len(self.coordSeq)):
			coordinate = []
			coordSeq = self.coordSeq[i]

			for j in range(len(coordSeq)):
				x, y, z = coordSeq[j]
				xx = int((x - self.initial_position[0])/pixelSpacing[0])
				yy = int((y - self.initial_position[1])/pixelSpacing[1])				
				zz = abs(z - self.initial_position[2])/pixelSpacing[2]
				#print(zz)
				#print(self.initial_position[2])   
				#print(pixelSpacing[2])
				coordinate.append([yy, xx, zz])
			self.coordSeqForImage.append(coordinate)

				