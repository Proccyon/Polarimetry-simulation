
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
    
def CreateAperture(Shape,x0,y0,R):
    Aperture = np.zeros(Shape)
    for y in range(Shape[0]):
        for x in range(Shape[1]):
            dr = np.sqrt((x-x0)**2 + (y-y0)**2)
            if(dr<=R):
                Aperture[y,x]=1
    
    return Aperture


#--/--Functions--/--#

#-----IrdisCalibrationClass-----#

class IrdisCalibration:
    
    def __init__(self,PolFileList,UnpolFileList,DarkFileF,DarkFileS,FlatFile):
        self.PolFileList = PolFileList
        self.UnpolFileList = UnpolFileList
        self.DarkFileF = DarkFileF
        self.DarkFileS = DarkFileS
        self.FlatFile = FlatFile
    
    def RunCalibration(self,UseAperture=True,ApertureSize=150):

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
        self.HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]
        self.PolDDImageArray,self.PolDSImageArray,self.PolDerList = self.CreateDoubleDifferenceImges(self.HwpTargetList,self.PolHwpListTotal,self.PolDerListTotal,self.PolImageListL,self.PolImageListR)
        self.UnpolDDImageArray,self.UnpolDSImageArray,self.UnpolDerList = self.CreateDoubleDifferenceImges(self.HwpTargetList,self.UnpolHwpListTotal,self.UnpolDerListTotal,self.UnpolImageListL,self.UnpolImageListR)

        print("Finding parameter values...")
        #Gets double difference values using aperatures
        self.PolParamValueArray = self.GetDoubleDifferenceValue(self.PolDDImageArray,self.PolDSImageArray,UseAperture,ApertureSize)
        self.UnpolParamValueArray = self.GetDoubleDifferenceValue(self.UnpolDDImageArray,self.UnpolDSImageArray,UseAperture,ApertureSize)

        print("Calculating fitted curve...")
        self.PolParamFitArray,self.PolFitDerList = self.FindFittedStokesParameters(SCExAO.BB_H_a,self.HwpTargetList,True)
        self.UnpolParamFitArray,self.UnpolFitDerList = self.FindFittedStokesParameters(SCExAO.BB_H_a,self.HwpTargetList,False)

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

        return np.array(DDImageArray),np.array(DSImageArray),np.array(DerList)


    #Uses aperatures to get a single value for the double differance
    def GetDoubleDifferenceValue(self,DDImageArray,DSImageArray,UseAperture=True,ApertureSize=150):
        if(UseAperture): #This only works for the mean, not for median
            Shape = DDImageArray[0][0].shape
            Aperture = CreateAperture(Shape,0.5*Shape[1],0.5*Shape[0],ApertureSize)
            return np.mean(Aperture*DDImageArray,axis=(2,3))/np.mean(Aperture*DSImageArray,axis=(2,3))
        
        else:
            return np.mean(DDImageArray,axis=(2,3))/np.mean(DSImageArray,axis=(2,3))


    def FindFittedStokesParameters(self,Model,HwpTargetList,UsePolarizer):

        FitDerList = np.linspace(-2,103.25*np.pi/180,200)    
        ParmFitValueArray = []
        S_In = np.array([1,0,0,0])

        for HwpTarget in HwpTargetList:
            HwpPlusTarget = HwpTarget[0]*np.pi/180
            HwpMinTarget = HwpTarget[1]*np.pi/180
            ParmFitValueList = []
            for FitDer in FitDerList:
                X_Matrix,I_Matrix = Model.ParameterMatrix(HwpPlusTarget,HwpMinTarget,FitDer,0,UsePolarizer,True)
                X_Out = np.dot(X_Matrix,S_In)[0]
                I_Out = np.dot(I_Matrix,S_In)[0]
                X_Norm = X_Out/I_Out
                ParmFitValueList.append(X_Norm)
        
            ParmFitValueArray.append(np.array(ParmFitValueList))
        
        return np.array(ParmFitValueArray),FitDerList


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

    
        #Shows one of the raw images. 
    def ShowDoubleDifferenceImage(self,ImageNumber,HwpNumber,FromPolarizedImages=True,DoubleSum=False):
        plt.figure()
        plt.xlabel("x(pixels)")
        plt.ylabel("y(pixels)")
        PlottedImage = []

        if(DoubleSum):
            if(FromPolarizedImages):
                PlottedImage = self.PolDSImageArray[HwpNumber][ImageNumber]
                plt.title("Polarized double sum image")
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


    def PlotStokesParameters(self,ColorList,FromPolarizedImages=True):
        
        plt.figure()
        plt.xlabel("Derotator angle")
        plt.ylabel("Normalized Stokes parameter (%)")
        plt.xticks(np.arange(-11.25,112.5,11.25))

        if(FromPolarizedImages):      
            plt.title("Stokes parameter vs derotator angle (polarizer)")           
            plt.yticks(np.arange(-120,120,20))
            plt.xlim(left=-1,right=91)
            plt.ylim(bottom=-110,top=110)

            
            for i in range(len(self.HwpTargetList)):
                #Plot data
                HwpPlusTarget = self.HwpTargetList[i][0]
                plt.scatter(self.PolDerList,100*self.PolParamValueArray[i],label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")

                #Plot fitted curve
                plt.plot(self.PolFitDerList*180/np.pi,self.PolParamFitArray[i]*100,color=ColorList[i])

        else:
            plt.title("Stokes parameter vs derotator angle (no polarizer)")           
            plt.yticks(np.arange(-1.2,1.2,0.2))
            plt.xlim(left=-1,right=102.25)
            plt.ylim(bottom=-1.1,top=1.1)

            
            for i in range(len(self.HwpTargetList)):
                #Plot data
                HwpPlusTarget = self.HwpTargetList[i][0]
                plt.scatter(self.UnpolDerList,100*self.UnpolParamValueArray[i],label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")

                #Plot fitted curve
                plt.plot(self.UnpolFitDerList*180/np.pi,self.UnpolParamFitArray[i]*100,color=ColorList[i])

        plt.grid(linestyle="--")
        plt.legend(fontsize=8)


    #---PlotFunctions---#

#--/--IrdisCalibrationClass--/--#

#-----Parameters-----#
Prefix = "C:/Users/Gebruiker/Desktop/BRP/CalibrationData/Internal Source/raw/irdis_internal_source_h/"
FileType=".fits"

PrefixUnpol = Prefix+"SPHERE_IRDIS_TEC165_0"
PrefixPol = Prefix+"SPHERE_IRDIS_TEC227_0"
DarkNameF = Prefix+"SPHERE_IRDIS_CAL_DARK164_0002"+FileType
DarkNameS = Prefix+"SPHERE_IRDIS_CAL_DARK164_0003"+FileType
FlatName = Prefix+"SPHERE_IRDIS_CAL_FLAT165_0002"+FileType

NumberListUnpol= np.arange(368,568,2)
NumberListPol = np.arange(400,664,2)

ApertureRadius = 30

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
IrdisCalibrationObject.RunCalibration(True,ApertureRadius)
IrdisCalibrationObject.PlotStokesParameters(["blue","lightblue","red","orange"],True)
IrdisCalibrationObject.PlotStokesParameters(["blue","lightblue","red","orange"],False)

plt.show()
#--/--Main--/--#

