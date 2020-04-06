
#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage, misc
from datetime import timedelta
import Methods as Mt
import SCExAO

#--/--Imports--/--#

#-----Class-----#

class SCExAO_Calibration():

    def __init__(self,PolFileList,RotationFile):
        self.PolFileList = PolFileList
        self.RotationFile = RotationFile

        self.ImageAngle = 27 #Angle with which the images seem to be rotated
        self.LeftX = 34 #Position in pixels of border on the left
        self.RightX = 166 #position in pixels of border on the right
        self.BottomY = 34 #Position in pixels of border on the top
        self.TopY = 166 #Position in pixels of border on the left
        self.MiddleX = 100 #Position in pixels of line that separates measurements
        self.PixelOffset = 5

        self.MaxTimeDifference = 5*60

        self.HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]


    def RunCalibration(self):
        print("Reading files...")
        self.PolImageList,self.PolLambdaList,self.PolTimeList = ReadCalibrationFiles(self.PolFileList)
        self.RotationTimeList,self.RotationImrList,self.RotationHwpList = ReadRotationFile(self.RotationFile)
        
        #print(self.PolTimeList/timedelta(seconds=1))

        #plt.scatter(range(len(self.PolTimeList)),self.PolTimeList/timedelta(minutes=1)-18*24*60,s=2)
        #plt.scatter(range(len(self.RotationTimeList)),self.RotationTimeList/timedelta(minutes=1)-18*24*60,s=2)
        #plt.axhline(y=24*60)
        #plt.show()

        print("Finding Imr and Hwp angles of calibration images...")
        self.PolImrList,self.PolHwpList,self.PolBadImageList = self.GetRotations(self.PolTimeList)
        self.PolImageList = self.PolImageList[self.PolBadImageList==False]
        self.PolLambdaList = self.PolLambdaList[self.PolBadImageList==False]

        print("Splitting calibration images...")
        self.PolImageListL,self.PolImageListR = self.SplitCalibrationImages(self.PolImageList)

        print("Creating double difference images...")
        self.PolDDImageArray,self.PolDSImageArray,self.PolImrArray = self.CreateHwpDoubleDifferenceImges(self.PolHwpList,self.PolImrList,self.PolImageListL,self.PolImageListR)
        
        print("Getting double difference value...")
        self.PolParamValueArray = self.GetDoubleDifferenceValue(self.PolDDImageArray,self.PolDSImageArray)
        

        #print(self.PolDDImageArray[0].shape)
        #print(self.PolDDImageArray[1].shape)
        #print(self.PolDDImageArray[2].shape)
        #print(self.PolDDImageArray[3].shape)
        #print(self.PolImrArray)

        #self.PlotParamValues(self.PolParamValueArray,0)

    #Rotates images and splits them into a left and right part
    def SplitCalibrationImages(self,ImageList):
        RotatedImageList = ndimage.rotate(ImageList,self.ImageAngle,reshape=False,axes=(2,3))

        ImageListL = RotatedImageList[:,:,self.BottomY+self.PixelOffset:self.TopY-self.PixelOffset,self.LeftX+self.PixelOffset:self.MiddleX-self.PixelOffset]
        ImageListR = RotatedImageList[:,:,self.BottomY+self.PixelOffset:self.TopY-self.PixelOffset,self.MiddleX+self.PixelOffset:self.RightX-self.PixelOffset]
        return ImageListL,ImageListR

    #Finds the Imr,Hwp angles per image from the list of rotations over time
    #BadImageList is a boolean array indicating images with invalid time
    def GetRotations(self,ImageTimeList):

        ImageImrList = [] #ImrAngle for each calibration image
        ImageHwpList = [] #HwpAngle for each calibration image
        BadImageList = [] #Boolean list of images without correct Imr,Hwp angles
        for i in range(len(ImageTimeList)):
            ImageTime = ImageTimeList[i]

            DeltaList = (self.RotationTimeList-ImageTime) / timedelta(seconds=1) #List of time differences in seconds

            TargetIndex = ArgMaxNegative(DeltaList)
            if(np.abs(DeltaList[TargetIndex]) <= self.MaxTimeDifference):
                BadImageList.append(False)
                ImageImrList.append(self.RotationImrList[TargetIndex])
                ImageHwpList.append(self.RotationHwpList[TargetIndex])
            else:
                BadImageList.append(True)

        return np.array(ImageImrList),np.array(ImageHwpList),np.array(BadImageList)

        #Creates double difference and sum images by combining images differing 45 degrees hwp angle
    def CreateHwpDoubleDifferenceImges(self,TotalHwpList,TotalImrList,ImageListL,ImageListR):

        DDImageArray = []
        DSImageArray = []
        ImrArray = []
        for HwpTarget in self.HwpTargetList:
            HwpPlusTarget = HwpTarget[0]
            HwpMinTarget = HwpTarget[1]
            ImrList = []
            DDImageList = []
            DSImageList = []
            for i in range(len(TotalHwpList)):
                if(TotalHwpList[i] == HwpMinTarget):
                    for j in range(len(TotalHwpList)):
                        if(TotalHwpList[j] == HwpPlusTarget and TotalImrList[i] == TotalImrList[j]):
                            ThetaImr = TotalImrList[i]
                            if(ThetaImr < 0):
                                ThetaImr += 180
                            ImrList.append(ThetaImr)
                            PlusDifference = ImageListL[j]-ImageListR[j]
                            MinDifference = ImageListL[i]-ImageListR[i]
                            PlusSum = ImageListL[j]+ImageListR[j]
                            MinSum = ImageListL[i]+ImageListR[i]
                            DDImage = 0.5*(PlusDifference - MinDifference) 
                            DSImage = 0.5*(PlusSum + MinSum)
                            DDImageList.append(DDImage)
                            DSImageList.append(DSImage)
                            break
                            
            DDImageArray.append(np.array(DDImageList))
            DSImageArray.append(np.array(DSImageList))
            ImrArray.append(np.array(ImrList))

        return np.array(DDImageArray),np.array(DSImageArray),np.array(ImrArray)

    #Uses aperatures to get a single value for the double differance
    def GetDoubleDifferenceValue(self,DDImageArray,DSImageArray):
        ParamValueArray = []
        for i in range(len(DDImageArray)):
            ParamValueArray.append(np.mean(DDImageArray[i],axis=(1,2)) / np.mean(DSImageArray[i],axis=(1,2)))
        
        return np.array(ParamValueArray)
        #for i in range(len(self.ApertureXList)):
        #    ApertureX = self.ApertureXList[i]
        #    ApertureY = self.ApertureYList[i] 
        #    Shape = DDImageArray[0][0].shape
        #    Aperture = CreateAperture(Shape,ApertureX,ApertureY,self.ApertureSize)
        #    ParamValue = np.median(DDImageArray[:,:,Aperture==1],axis=2) / np.median(DSImageArray[:,:,Aperture==1],axis=2)
        #    ParamValueArray.append(ParamValue)

        #return np.array(ParamValueArray)

    def PlotParamValues(self,ParamValueArray,LambdaNumber):
        plt.figure()
        plt.ylabel("Normalized Stokes parameter (%)")
        plt.xticks(np.arange(45,112.5,7.5))

      
        plt.title("Stokes parameter vs Imr angle (polarizer)")           
        plt.xlabel("Imr angle(degrees)")
        plt.yticks(np.arange(-120,120,20))
        plt.xlim(left=44,right=113.5)
        plt.ylim(bottom=-100,top=100)
        plt.axhline(y=0,color="black")

        for i in range(len(self.HwpTargetList)):
            #Plot data
            HwpPlusTarget = self.HwpTargetList[i][0]
            ParamValueList = 100*self.PolParamValueArray[i][:,LambdaNumber]
            plt.scatter(self.PolImrArray[i],ParamValueList,label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")
        
        plt.grid(linestyle="--")
        plt.legend(fontsize=8)


    
#--/--Class--/--#

#-----Functions-----#

#---ReadInFunctions---#

#Gets images and header data from fits files
def ReadCalibrationFiles(FileList):
    LambdaList = []
    ImageList = []
    TimeList = []
    for File in FileList:
        Header = File[0].header
        Image = File[1].data
        RawHeader = File[3].header #Not currently used, not sure what to do with this
        Lambda = Header['lam_min']*np.exp(np.arange(Image.shape[0])*Header['dloglam']) #This gets the wavelength...
        LambdaList.append(Lambda)
        ImageList.append(Image)
        Days = float(Header["UTC-Date"][-2:])

        TimeRow = Header["UTC-Time"].split(":")
        #Converts time to the timedelta data type
        TimeList.append(timedelta(hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2]),days=Days))
    
    return np.array(ImageList),np.array(LambdaList),np.array(TimeList)

#Reads the file with rotations for each date,time
def ReadRotationFile(RotationFile):

    TimeList = []
    ImrAngleList = []
    HwpAngleList = []

    for Row in RotationFile:
        RowList = Row.split(" ") 

        TimeRow = RowList[1].split(":")
        TimeList.append(timedelta(days=float(RowList[0][-2:]),hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2])))

        ImrAngleList.append(float(RowList[2]))
        HwpAngleList.append(float(RowList[3][:-1]))#Also removes the /n on the end with the [:-1]

    return np.array(TimeList),np.array(ImrAngleList),np.array(HwpAngleList)



#-/-ReadInFunctions-/-#

#---OtherFunctions---#
#Finds the index of the highest(closest to zero) negative number in a list
def ArgMaxNegative(List):
    List = List*(List<=0) - List*(List>0)*1E6
    return np.argmax(List)

#-/-OtherFunctions-/-#

#-----Parameters-----#

#Path to calibration files
PolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_pol_source/CRSA000"
UnpolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA000"
RotationPath = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/RotationsChanged.txt"

#Positions in rotated image
ImageAngle = 27
LeftX = 34 #Position in pixels of border on the left
RightX = 166 #position in pixels of border on the right
BottomY = 34 #Position in pixels of border on the top
TopY = 166 #Position in pixels of border on the left
MiddleX = 100 #Position in pixels of line that separates measurements
PixelOffset = 5

PolNumberList = np.arange(59565,59905)
UnpolNumberList = np.arange(59559,59565)

MaxTimeDifference = 5*60 #Max time difference between hwp,imr angle switch and image measurement in seconds

HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]
ColorList = ["blue","lightblue","red","orange"]

LambdaNumber = 0

#--/--Parameters--/--#

#-----Main-----#

#Get the file with rotations over time
RotationFile = open(RotationPath, "r")

#Get the polarized calibration images
PolFileList = []
for PolNumber in PolNumberList:
    PolPath = PolPrefix + str(PolNumber) + "_cube.fits"
    PolFile = fits.open(PolPath)
    PolFileList.append(PolFile)

SCExAO_CalibrationObject = SCExAO_Calibration(PolFileList,RotationFile)
SCExAO_CalibrationObject.RunCalibration()

plt.show()


#--/--Main--/--#



