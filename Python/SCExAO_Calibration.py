
#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage, misc
from datetime import timedelta
import Methods as Mt
import SCExAO

#--/--Imports--/--#

#-----Functions-----#

#Gets images and header data from fits files
def ReadCalibrationFile(Prefix,NumberList):
    LambdaList = []
    ImageList = []
    DateList = []
    TimeList = []
    for Number in NumberList:
        Path = Prefix + str(Number) + "_cube.fits"
        HduList = fits.open(Path)
        Header = HduList[0].header
        Image = HduList[1].data
        RawHeader = HduList[3].header #Not currently used, not sure what to do with this
        Lambda = Header['lam_min']*np.exp(np.arange(Image.shape[0])*Header['dloglam']) #This gets the wavelength...
        LambdaList.append(Lambda)
        ImageList.append(Image)
        DateList.append(Header["UTC-Date"])

        TimeRow = Header["UTC-Time"].split(":")
        TimeList.append(timedelta(hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2])))
    
    return np.array(ImageList),np.array(LambdaList),np.array(DateList),np.array(TimeList)

#Rotates images and splits them into a left and right part
def SplitCalibrationImages(ImageList,ImageAngle,LeftX,RightX,BottomY,TopY,MiddleX,PixelOffset):
    RotatedImageList = ndimage.rotate(ImageList,ImageAngle,reshape=False,axes=(2,3))

    plt.figure()
    plt.imshow(RotatedImageList[0][0],vmin=500,vmax=1.5E3,cmap="gist_gray")
    plt.xlabel("x(pixels)")
    plt.ylabel("y(pixels)")
    plt.colorbar()

    ImageListL = RotatedImageList[:,:,BottomY+PixelOffset:TopY-PixelOffset,LeftX+PixelOffset:MiddleX-PixelOffset]
    ImageListR = RotatedImageList[:,:,BottomY+PixelOffset:TopY-PixelOffset,MiddleX+PixelOffset:RightX-PixelOffset]
    return ImageListL,ImageListR

#Reads the file with rotations for each date,time
def ReadRotationFile(RotationPath):

    DateList = []
    TimeList = []
    ImrAngleList = []
    HwpAngleList = []

    File = open(RotationPath, "r")
    for Row in File:
        RowList = Row.split(" ") 

        TimeRow = RowList[1].split(":")
        TimeList.append(timedelta(hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2])))
        DateList.append(RowList[0])
        ImrAngleList.append(float(RowList[2]))
        HwpAngleList.append(float(RowList[3][:-1]))#Also removes the /n on the end with the [:-1]

    return np.array(DateList),np.array(TimeList),np.array(ImrAngleList),np.array(HwpAngleList)

#Finds the index of the highest(closest to zero) negative number in a list
def ArgMaxNegative(List):
    List = List*(List<=0) - List*(List>0)*1E6
    return np.argmax(List)

#Finds the Imr,Hwp angles per image from the list of rotations over time
def GetRotations(RotationDateList,RotationTimeList,RotationImrList,RotationHwpList,ImageDateList,ImageTimeList,MaxTimeDifference):

    ImageImrList = [] #ImrAngle for each calibration image
    ImageHwpList = [] #HwpAngle for each calibration image
    BadImageList = [] #Boolean list of images without correct Imr,Hwp angles
    for i in range(len(ImageDateList)):
        ImageDate = ImageDateList[i]
        ImageTime = ImageTimeList[i]
        DeltaList = (RotationTimeList-ImageTime)/timedelta(seconds=1) #List of time differences in seconds
        DeltaList = DeltaList*(ImageDate==RotationDateList)+1E6*(ImageDate!=RotationDateList) #Remove entries on incorrect day
        TargetIndex = ArgMaxNegative(DeltaList)
        if(np.abs(DeltaList[TargetIndex]) <= MaxTimeDifference):
            BadImageList.append(False)
            ImageImrList.append(RotationImrList[TargetIndex])
            ImageHwpList.append(RotationHwpList[TargetIndex])
        else:
            BadImageList.append(True)

    return np.array(ImageImrList),np.array(ImageHwpList),np.array(BadImageList)

def FindDoubleDifference(HwpPlusTarget,HwpMinTarget,TotalHwpList,TotalImrList,ImageListL,ImageListR,LambdaNumber):

    ImrList = []
    DoubleDifferenceList = []
    for i in range(len(TotalHwpList)):
        if(TotalHwpList[i] == HwpPlusTarget):
            for j in range(len(TotalHwpList)):
                if(TotalHwpList[j] == HwpMinTarget and TotalImrList[i] == TotalImrList[j]):
                    ThetaImr = TotalImrList[i]
                    if(ThetaImr < 0):
                        ThetaImr += 180
                    ImrList.append(ThetaImr)
                    PlusDifference = ImageListL[i]-ImageListR[i]
                    MinDifference = ImageListL[j]-ImageListR[j]
                    PlusSum = ImageListL[i]+ImageListR[i]
                    MinSum = ImageListL[j]+ImageListR[j]
                    DDImage = (PlusDifference - MinDifference) / (PlusSum+MinSum)
                    DoubleDifference = np.mean(DDImage[LambdaNumber])
                    DoubleDifferenceList.append(DoubleDifference)

    return DoubleDifferenceList,ImrList

def PlotDoubleDifference(HwpTargetList,TotalHwpList,TotalImrList,ImageListL,ImageListR,ColorList,Title,LambdaList,LambdaNumber):

    #BB_H = SCExAO.BB_H_a #Matrix model
    #FitDerList = np.linspace(-0.1*np.pi,0.6*np.pi,200)
    #S_In = np.array([1,0.00948,0.000406,0])

    plt.figure()   
    plt.xticks(np.arange(37.5,135,7.5))
    plt.xlim(left=43,right=129.50)
    plt.ylim(bottom=-100,top=100)

    plt.title(Title+"(Lambda="+str(int(LambdaList[0][LambdaNumber]))+"nm)")
    plt.xlabel("IMR angle (degree)")
    plt.ylabel("Normalized Stokes parameter (%)")

    plt.axhline(y=0,color="black")
        
    for i in range(len(HwpTargetList)):
            
        #Plot measured data points
        HwpTarget = HwpTargetList[i]
        HwpPlusTarget = HwpTarget[0]
        HwpMinTarget = HwpTarget[1]
            
        DDList,ImrList = FindDoubleDifference(HwpPlusTarget,HwpMinTarget,TotalHwpList,TotalImrList,ImageListL,ImageListR,LambdaNumber)

        plt.scatter(ImrList,np.array(DDList)*100,label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=ColorList[i],s=18,edgecolors="black")

         #Plot fitted curve
         #ParmFitValueList = []
        #for FitDer in FitDerList:
        #    X_Matrix,I_Matrix = BB_H.ParameterMatrix(HwpPlusTarget*np.pi/180,HwpMinTarget*np.pi/180,FitDer,0,Polarizer)
        #    X_Out = np.dot(X_Matrix,S_In)[0]
        #    I_Out = np.dot(I_Matrix,S_In)[0]
        #    X_Norm = X_Out/I_Out
        #    ParmFitValueList.append(X_Norm)

        #plt.plot(FitDerList*180/np.pi,np.array(ParmFitValueList)*100,color=ColorList[i])

    plt.grid(linestyle="--")
    plt.legend(fontsize=8)

    plt.figure()

#--/--Functions--/--#

#-----Parameters-----#

#Path to calibration files
PolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_pol_source/CRSA000"
UnpolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA000"
RotationPath = "C:/Users/Gebruiker/Desktop\BRP/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/RotationsChanged.txt"

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

print("#-----ReadingPolData-----#")
PolImageList,PolLambdaList,PolDateList,PolTimeList = ReadCalibrationFile(PolPrefix,PolNumberList)

print("#-----ReadingUnpolData-----#")
UnpolImageList,UnpolLambdaList,UnpolDateList,UnpolTimeList = ReadCalibrationFile(UnpolPrefix,UnpolNumberList)

print("#-----ReadingRotationFile-----#")
RotationDateList,RotationTimeList,RotationImrList,RotationHwpList = ReadRotationFile(RotationPath)

print("#-----FindPolRotations-----#")
PolImrList,PolHwpList,PolBadImageList = GetRotations(RotationDateList,RotationTimeList,RotationImrList,RotationHwpList,PolDateList,PolTimeList,MaxTimeDifference)
PolImageList = PolImageList[PolBadImageList==False]
PolLambdaList = PolLambdaList[PolBadImageList==False]

print("#-----FindUnpolRotations-----#")
UnpolImrList,UnpolHwpList,UnpolBadImageList = GetRotations(RotationDateList,RotationTimeList,RotationImrList,RotationHwpList,UnpolDateList,UnpolTimeList,MaxTimeDifference)
UnpolImageList = UnpolImageList[UnpolBadImageList==False]
UnpolLambdaList = UnpolLambdaList[UnpolBadImageList==False]

#plt.figure()
#plt.imshow(PolImageList[0][0],vmin=500,vmax=1.5E3,cmap="gist_gray")
#plt.xlabel("x(pixels)")
#plt.ylabel("y(pixels)")
#plt.colorbar()

print("#-----SplitPolImages-----#")
PolImageListL,PolImageListR = SplitCalibrationImages(PolImageList,ImageAngle,LeftX,RightX,BottomY,TopY,MiddleX,PixelOffset)

#plt.figure()
#plt.imshow(PolImageListL[0][0],vmin=500,vmax=1.5E3,cmap="gist_gray")
#plt.xlabel("x(pixels)")
#plt.ylabel("y(pixels)")
#plt.colorbar()

#plt.figure()
#plt.imshow(PolImageListR[0][0],vmin=500,vmax=1.5E3,cmap="gist_gray")
#plt.xlabel("x(pixels)")
#plt.ylabel("y(pixels)")
#plt.colorbar()

#print("#-----SplitUnpolImages-----#")
#UnpolImageListL,UnpolImageListR = SplitCalibrationImages(UnpolImageList,ImageAngle,LeftX,RightX,BottomY,TopY,MiddleX,PixelOffset)

#print("#-----PlotPolDoubleDifference-----#")
#PlotDoubleDifference(HwpTargetList,PolHwpList,PolImrList,PolImageListL,PolImageListR,ColorList,"Stokes parameters for polarized calibration data",PolLambdaList,LambdaNumber)

#print("#-----PlotUnpolDoubleDifference-----#")
#PlotDoubleDifference(HwpTargetList,UnpolHwpList,UnpolImrList,UnpolImageListL,UnpolImageListR,ColorList,"Stokes parameters for unpolarized calibration data",PolLambdaList,LambdaNumber)

plt.show()
#--/--Main--/--#



