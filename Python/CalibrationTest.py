import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)

#-----Functions-----#

def CircleMean(Theta1,Theta2):
    return np.arctan( (np.sin(Theta1)+np.sin(Theta2)) / (np.cos(Theta1)+np.cos(Theta2)) )
    
def FindHeaderParameters(File,PrintParameters=False):    
    Header = File[0].header
    
    Der1 = Header["ESO INS4 DROT2 BEGIN"]*np.pi/180
    Der2 = Header["ESO INS4 DROT2 END"]*np.pi/180
    DerAngle = np.round(CircleMean(Der1,Der2)*180/np.pi,2)
    
    Hwp1 = (Header["ESO INS4 DROT3 BEGIN"]-152.15)*np.pi/180
    Hwp2 = (Header["ESO INS4 DROT3 END"]-152.15)*np.pi/180
    HwpAngle = np.round(CircleMean(Hwp1,Hwp2)*180/np.pi,2)
    
    Polarizer = Header["ESO INS4 OPTI7 NAME"]
    ExpTime = Header["EXPTIME"]
    
    if(PrintParameters):
        print("DerAngle = "+str(DerAngle))
        print("HwpAngle = "+str(HwpAngle))
        print("Polarizer = "+str(Polarizer))
        print("ExpTime = "+str(ExpTime))
    
    return [DerAngle,HwpAngle,Polarizer,ExpTime]
    
    
#--/--Functions--/--#



#-----Parameters-----#
Prefix = "C:/Users/Jasper/Desktop/BRP/CalibrationData/Internal Source/raw/irdis_internal_source_h/"
FileType=".fits"

Prefix1 = Prefix+"SPHERE_IRDIS_TEC165_0"
Prefix2 = Prefix+"SPHERE_IRDIS_TEC227_0"
DarkName = Prefix+"SPHERE_IRDIS_CAL_DARK164_0002"+FileType
FlatName = Prefix+"SPHERE_IRDIS_CAL_FLAT165_0002"+FileType
BlaName = Prefix1+"368"+FileType

#--/--Parameters--/--#

#-----ReadData-----#

NumberList1 = np.arange(368,566,2)
NumberList2 = np.arange(400,662,2)

DerList1 = []
DerList2 = []

HwpList1 = []
HwpList2 = []

ExpTimeList1 = []
ExpTimeList2 = []

ImageList1L = []
ImageList1R = []

ImageList2L = []
ImageList2R = []

print("#-----Dark-----#")
DarkFile = fits.open(DarkName)
DarkExpTime = FindHeaderParameters(DarkFile,False)[3]
DarkImage = DarkFile[0].data[1] / DarkExpTime


print("#-----Flat-----#")
FlatFile = fits.open(FlatName)
FlatExpTime = FindHeaderParameters(FlatFile,False)[3]
FlatImage = FlatFile[0].data[0] / FlatExpTime  

print("#-----Data1-----#")
for Number in NumberList1:
    Name = Prefix1+str(Number)+FileType
    File = fits.open(Name)
    
    DerAngle,HwpAngle,Polarizer,ExpTime = FindHeaderParameters(File)
    DerList1.append(DerAngle)
    HwpList1.append(HwpAngle)
    ExpTimeList1.append(ExpTime)
    
    Image = (File[0].data[0]/ExpTime-DarkImage) / (FlatImage-DarkImage)
    ImageL = Image[15:1024, 36:933]
    ImageR = Image[5:1018, 1062:1958]
    
    ImageList1L.append(ImageL)
    ImageList1R.append(ImageR)
    File.close()
    
print("#-----Data2-----#")
for Number in NumberList2:
    Name = Prefix2+str(Number)+FileType
    File = fits.open(Name)
    
    DerAngle,HwpAngle,Polarizer,ExpTime = FindHeaderParameters(File)
    DerList2.append(DerAngle)
    HwpList2.append(HwpAngle)
    ExpTimeList2.append(ExpTime)
    
    Image = (File[0].data[0]/ExpTime-DarkImage) / (FlatImage-DarkImage)
    ImageL = Image[15:1024, 36:933]
    ImageR = Image[5:1018, 1062:1958]
    
    ImageList2L.append(ImageL)
    ImageList2R.append(ImageR)
    File.close()
    
 
    
#--/--ReadData--/--#

#-----ShowData-----#

plt.imshow(np.log(ImageList1L[0]),vmin=-1.1,vmax=-0.9)
plt.title("Log image of left dark/flat reduced image(no polarizer)")
plt.colorbar()
plt.show()


#--/--ShowData--/--#