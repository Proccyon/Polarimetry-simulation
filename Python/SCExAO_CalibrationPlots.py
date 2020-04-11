
#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage, misc
from datetime import timedelta
import Methods as Mt
import SCExAO_Model
import SCExAO_CalibrationMain

#--/--Imports--/--#

#-----PlotFunctions-----#

def PlotParamValues(self,LambdaNumber,PlotIrdisCurve=False):
    plt.figure()
    plt.ylabel("Normalized Stokes parameter (%)")
    plt.xticks(np.arange(45,135,7.5))

    Lambda = self.PolLambdaList[0][LambdaNumber]
    plt.title("Stokes parameter vs Imr angle (polarizer)(Lambda="+str(int(Lambda))+"nm")           
    plt.xlabel("Imr angle(degrees)")
    plt.yticks(np.arange(-120,120,20))
    plt.xlim(left=44,right=128.5)
    plt.ylim(bottom=-100,top=100)
    plt.axhline(y=0,color="black")

    Model = SCExAO_Model.BB_H

    FitDerList = np.linspace(-43*np.pi/180,130*np.pi/180,200)
    S_In = np.array([1,0,0,0])       


    for i in range(len(self.HwpTargetList)):
        #Plot data
        HwpPlusTarget = self.HwpTargetList[i][0]
        HwpMinTarget = self.HwpTargetList[i][1]
        ParamValueList = 100*self.PolParamValueArray[i][:,LambdaNumber]
        plt.scatter(self.PolImrArray[i],ParamValueList,label="HwpPlus = "+str(HwpPlusTarget),zorder=100,color=self.ColorList[i],s=18,edgecolors="black")

        if(PlotIrdisCurve):
            ParamFitValueList = []
            for FitDer in FitDerList:
                X_Matrix,I_Matrix = Model.ParameterMatrix(HwpPlusTarget,HwpMinTarget,FitDer,0,True,True,False)
                X_Out = np.dot(X_Matrix,S_In)[0]
                I_Out = np.dot(I_Matrix,S_In)[0]
                X_Norm = X_Out/I_Out
                ParamFitValueList.append(X_Norm)
        
            plt.plot(FitDerList*180/np.pi,np.array(ParamFitValueList)*100,color=self.ColorList[i])

    plt.grid(linestyle="--")
    plt.legend()
    
def PlotPolarizationDegree(self,LambdaNumber):
    plt.figure()
    plt.ylabel("Degree of linear polarization(%)")
    plt.xticks(np.arange(45,135,7.5))

    Lambda = self.PolLambdaList[0][LambdaNumber]  
    plt.title("Pol degree vs Imr angle (polarizer)(Lambda="+str(int(Lambda))+"nm)")           
    plt.xlabel("Imr angle(degrees)")
    plt.yticks(np.arange(0,100,10))
    plt.xlim(left=44,right=128.5)
    plt.ylim(bottom=0,top=100)

    PolDegree = np.sqrt(self.PolParamValueArray[0][:,LambdaNumber]**2+self.PolParamValueArray[2][:,LambdaNumber]**2)
    #PolDegree = np.sqrt(self.PolParamValueArray[1][:,LambdaNumber]**2+self.PolParamValueArray[3][:,LambdaNumber]**2)

    plt.scatter(self.PolImrArray[0],100*PolDegree,zorder=100,s=18,edgecolors="black")
        
    plt.grid(linestyle="--")

def ShowDoubleDifferenceImage(self,HwpNumber,DerNumber,LambdaNumber,DoSum=False):

    plt.figure()
    if(DoSum):
        plt.title("Double sum image")
        Image = self.PolDSImageArray[HwpNumber][DerNumber][LambdaNumber]

    else:
        plt.title("Double difference image")
        Image = self.PolDDImageArray[HwpNumber][DerNumber][LambdaNumber]

    plt.imshow(Image,vmin=np.mean(Image)*0.6,vmax=np.mean(Image)*1.4)
    plt.colorbar()
    

#--/--PlotFunctions-----#

#-----SetFunctions-----#
SCExAO_CalibrationMain.SCExAO_Calibration.PlotParamValues = PlotParamValues
SCExAO_CalibrationMain.SCExAO_Calibration.ShowDoubleDifferenceImage = ShowDoubleDifferenceImage
SCExAO_CalibrationMain.SCExAO_Calibration.PlotPolarizationDegree = PlotPolarizationDegree
#--/--SetFunctions--/--#

#-----Main-----#

SCExAO = SCExAO_CalibrationMain.SCExAO_CalibrationObject
SCExAO.PlotParamValues(10,True)
#SCExAO.ShowDoubleDifferenceImage(0,0,0,True)
#SCExAO.PlotPolarizationDegree(2)
plt.show()



#--/--Main--/--#