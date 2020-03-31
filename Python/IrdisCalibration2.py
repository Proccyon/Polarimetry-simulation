
#-----Imports-----#
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import warnings

import Methods as Mt
import SCExAO
#--/--Imports--/--#

warnings.filterwarnings('ignore', category=UserWarning, append=True)

#-----Functions-----#

#Takezs the mean of two angles, ex. makes sure (7/4pi,0) goes well
def CircleMean(Theta1,Theta2):
    return np.arctan( (np.sin(Theta1)+np.sin(Theta2)) / (np.cos(Theta1)+np.cos(Theta2)) )
    
#--/--Functions--/--#

#-----IrdisCalibrationClass-----#

class IrdisCalibration:
    
    def __init__(self,PolFileList,UnpolFileList,DarkFileF,DarkFileS,FlatFile):
        self.PolFileList = PolFileList
        self.UnpolFileList = UnpolFileList
        self.DarkFileF = DarkFileF
        self.DarkFileS = DarkFileS
        self.FlatFile = FlatFile
    
    def RunCalibration(self):

        print("Reading files...")
        #Get the images and header parameters of calibration images
        self.RawPolImageList,self.PolDerListTotal,self.PolHwpListTotal,self.PolExpTimeList = self.ReadFileList(PolFileList)
        self.RawUnpolImageList,self.UnpolDerListTotal,self.UnpolHwpListTotal,self.UnpolExpTimeList = self.ReadFileList(UnpolFileList)

        #Get images and header parameters of dark images
        #DarkImageF is the dark image subtracted from the flat, DarkImageS is for the calibration images
        #The dark hwp and der angles are not used for anything
        DarkImageListF,DarkDerF,DarkHwpF,self.DarkExpTimeF = self.ReadFile(DarkFileF)
        self.DarkImageF = np.mean(DarkImageListF,axis=0)
        DarkImageListS,DarkDerS,DarkHwpS,self.DarkExpTimeS = self.ReadFile(DarkFileS)
        self.DarkImageS = np.mean(DarkImageListS,axis=0)

        #Get flat image and header parameters
        self.FlatImage,FlatDer,FlatHwp,self.FlatExpTime = self.ReadFile(FlatFile)

        print("Calibrating files...")
        #Calibrate for noise using the darks and flat
        self.PolImageList = self.CalibrateNoise(self.RawPolImageList,self.PolExpTimeList,self.FlatImage,self.FlatExpTime,self.DarkImageS,self.DarkImageF)
        self.UnpolImageList = self.CalibrateNoise(self.RawUnpolImageList,self.UnpolExpTimeList,self.FlatImage,self.FlatExpTime,self.DarkImageS,self.DarkImageF)

        print("Splitting files...")
        #Splits the images in two
        self.PolImageListL,self.PolImageListR = self.SplitImageList(self.PolImageList)
        self.UnpolImageListL,self.UnpolImageListR = self.SplitImageList(self.UnpolImageList)

        print("Creating double difference images...")
        #Creates the double difference images
        HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]
        self.PolDDImageArray,self.PolDSImageArray,self.PolDerList = self.CreateDoubleDifferenceImges(HwpTargetList,self.PolHwpListTotal,self.PolDerListTotal,self.PolImageListL,self.PolImageListR)
        self.UnpolDDImageArray,self.UnpolDSImageArray,self.UnpolDerList = self.CreateDoubleDifferenceImges(HwpTargetList,self.UnpolHwpListTotal,self.UnpolDerListTotal,self.UnpolImageListL,self.UnpolImageListR)

        #Gets double difference values using aperatures
        self.PolDDArray = self.GetDoubleDifferenceValue(self.PolDDImageArray,self.PolDSImageArray)
        self.UnpolDDArray = self.GetDoubleDifferenceValue(self.UnpolDDImageArray,self.UnpolDSImageArray)

    #---ReadFunctions---#
    def ReadFile(self,File):    
        Header = File[0].header
        RawImage = File[0].data[0]

        Der1 = Header["ESO INS4 DROT2 BEGIN"]*np.pi/180
        Der2 = Header["ESO INS4 DROT2 END"]*np.pi/180
        DerAngle = np.round(CircleMean(Der1,Der2)*180/np.pi,2)
        
        Hwp1 = (Header["ESO INS4 DROT3 BEGIN"]-152.15)*np.pi/180
        Hwp2 = (Header["ESO INS4 DROT3 END"]-152.15)*np.pi/180
        HwpAngle = np.round(CircleMean(Hwp1,Hwp2)*180/np.pi,2)
       
        ExpTime = Header["EXPTIME"]

        return RawImage,DerAngle,HwpAngle,ExpTime

    def ReadFileList(self,FileList):

        RawImageList = []
        Derlist = []
        HwpList = []
        ExpTimeList = []
        for File in FileList:
            RawImage,DerAngle,HwpAngle,ExpTime = self.ReadFile(File)
            RawImageList.append(RawImage)
            Derlist.append(DerAngle)
            HwpList.append(HwpAngle)
            ExpTimeList.append(ExpTime)
        
        return np.array(RawImageList),np.array(Derlist),np.array(HwpList),np.array(ExpTimeList)

    #-/-ReadFunctions-/-#

    #---EssentialFunctions---#
    #CHANGE LATER                                            tHIS|
    def CalibrateNoise(self,RawImageList,ImageExpTimeList,FlatImage,FlatExpTime,DarkImageS,DarkImageF):
        return np.mean(FlatImage)*(FlatExpTime /ImageExpTimeList[0])*(RawImageList - DarkImageS) / (FlatImage - DarkImageF)

    def SplitImageList(self,ImageList):
        ImageL = ImageList[:,11:1024, 36:932]
        ImageR = ImageList[:,5:1018, 1062:1958]
        return ImageL,ImageR

    def CreateDoubleDifferenceImges(self,HwpTargetList,TotalHwpList,TotalDerList,ImageListL,ImageListR):

        DDImageArray = []
        DSImageArray = []
        for HwpTarget in HwpTargetList:
            HwpPlusTarget = HwpTarget[0]
            HwpMinTarget = HwpTarget[1]
            DerList = []
            DDImageList = []
            DSImageList = []
            for i in range(len(TotalHwpList)):
                if(TotalHwpList[i] == HwpMinTarget):
                    for j in range(len(TotalHwpList)):
                        if(TotalHwpList[j] == HwpPlusTarget and TotalDerList[i] == TotalDerList[j]):
                            ThetaDer = TotalDerList[i]
                            if(ThetaDer < 0):
                                ThetaDer += 180
                            DerList.append(ThetaDer)
                            PlusDifference = ImageListL[i]-ImageListR[i]
                            MinDifference = ImageListL[j]-ImageListR[j]
                            PlusSum = ImageListL[i]+ImageListR[i]
                            MinSum = ImageListL[j]+ImageListR[j]
                            DDImage = PlusDifference - MinDifference 
                            DSImage = PlusSum + MinSum
                            DDImageList.append(DDImage)
                            DSImageList.append(DSImage)
                            break
                            
            DDImageArray.append(np.array(DDImageList))
            DSImageArray.append(np.array(DSImageList))

        return np.array(DDImageArray),np.array(DSImageArray),np.array(DerList)


    #Uses aperatures to get a single value for the double differance
    def GetDoubleDifferenceValue(self,DDImageArray,DSImageArray):
        return np.mean(DDImageArray,axis=0)/np.mean(DSImageArray,axis=0)

    #-/-EssentialFunctions-/-#


    #---PlotFunctions---#
    #Shows one of the raw images. 
    def ShowRawImage(self,ImageNumber,FromPolarizedImages=True):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        
        if(FromPolarizedImages):
            plt.imshow(self.RawPolImageList[ImageNumber])
            plt.title("Raw polarized calibration image")
        else:
            plt.imshow(self.RawUnpolImageList[ImageNumber])
            plt.title("Raw unpolarized calibration image")

        plt.colorbar()
        plt.show()

    #Shows one of the raw images. 
    def ShowCalibratedImage(self,ImageNumber,FromPolarizedImages=True):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        
        if(FromPolarizedImages):
            plt.imshow(self.PolImageList[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.title("Dark,Flat calibrated polarized calibration image")
        else:
            plt.imshow(self.UnpolImageList[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.title("Dark,Flat calibrated unpolarized calibration image")

        plt.colorbar()
        plt.show()
            
    #Shows one of the raw images. 
    def ShowLeftRightImage(self,ImageNumber,FromPolarizedImages=True):

        if(FromPolarizedImages):
            plt.figure()
            plt.title("Left polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.PolImageListL[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

            plt.figure()
            plt.title("Right polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.PolImageListR[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

            
        else:
            plt.figure()
            plt.title("Left polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.UnpolImageListL[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

            plt.figure()
            plt.title("Right polarized image")
            plt.xlabel("x(pixels)")
            plt.ylabel("y(pixels)")
            plt.imshow(self.UnpolImageListR[ImageNumber],vmin=4.8E3,vmax=5.2E3)
            plt.colorbar()

        plt.show()
    
        #Shows one of the raw images. 
    def ShowDoubleDifferenceImage(self,ImageNumber,HwpNumber,FromPolarizedImages=True,DoubleSum=False):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        PlottedImage = []

        if(DoubleSum):
            if(FromPolarizedImages):
                PlottedImage = self.PolDSImageArray[HwpNumber][ImageNumber]
                plt.title("Polarized double sum image",vmin=npmean)
            else:
                PlottedImage = self.UnpolDSImageArray[HwpNumber][ImageNumber]
                plt.title("Unpolarized double sum image")
        else:
            if(FromPolarizedImages):
                PlottedImage = self.PolDDImageArray[HwpNumber][ImageNumber]
                plt.title("Polarized double difference image")
            else:
                PlottedImage = self.UnpolDDImageArray[HwpNumber][ImageNumber]
                plt.title("Unpolarized double difference image")

        plt.imshow(PlottedImage,vmin=np.mean(PlottedImage)/1.2,vmax=np.mean(PlottedImage)*1.2)
        plt.colorbar()
        plt.show()

    #---PlotFunctions---#

#--/--IrdisCalibrationClass--/--#

#-----Parameters-----#
Prefix = "C:/Users/Gebruiker/Desktop/BRP/CalibrationData/Internal Source/raw/irdis_internal_source_h/"
FileType=".fits"

PrefixPol = Prefix+"SPHERE_IRDIS_TEC165_0"
PrefixUnpol = Prefix+"SPHERE_IRDIS_TEC227_0"
DarkNameF = Prefix+"SPHERE_IRDIS_CAL_DARK164_0002"+FileType
DarkNameS = Prefix+"SPHERE_IRDIS_CAL_DARK164_0003"+FileType
FlatName = Prefix+"SPHERE_IRDIS_CAL_FLAT165_0002"+FileType

Altitude = (1/8)*np.pi
NumberListPol= np.arange(368,568,2)
NumberListUnpol = np.arange(400,664,2)

#--/--Parameters--/--#

#-----FindFiles-----#
DarkFileF = fits.open(DarkNameF)
DarkFileS = fits.open(DarkNameS)
FlatFile = fits.open(FlatName)

PolFileList  = []
for Number in NumberListPol:
    PolFile = fits.open(PrefixPol+str(Number)+".fits")
    PolFileList.append(PolFile)

UnpolFileList  = []
for Number in NumberListUnpol:
    UnpolFile = fits.open(PrefixUnpol+str(Number)+".fits")
    UnpolFileList.append(UnpolFile)

#--/--FindFiles--/--#

#-----Main-----#

IrdisCalibrationObject = IrdisCalibration(PolFileList,UnpolFileList,DarkFileF,DarkFileS,FlatFile)
IrdisCalibrationObject.RunCalibration()
#IrdisCalibrationObject.ShowDoubleDifferenceImage(0,0,False,False)
#--/--Main--/--#

