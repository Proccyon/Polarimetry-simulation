import numpy as np
import matplotlib.pyplot as plt
import Methods as Mt

#-----FilterClass-----#

#The optical system of sphere for a specific wavelength
class Filter():
    
    def __init__(self,Name,eHwp,RHwp,DeltaHwp,eDer,RDer,DeltaDer,d,DeltaCal):
        self.Name = Name #Name of filter
        self.eHwp = eHwp #Diattenuation of half-waveplate
        self.RHwp = RHwp *np.pi/180 #Retardance of half-waveplate
        self.DeltaHwp = DeltaHwp*np.pi/180 #Offset of half-waveplate
        self.eDer = eDer #Diattenuation of derotator
        self.RDer = RDer*np.pi/180 #Retardance of derotator
        self.DeltaDer = DeltaDer*np.pi/180 #Offset of derotator
        self.d = d #Diattenuation of polarizers
        self.DeltaCal = DeltaCal*np.pi/180 #Offset of calibration polarizer
        self.Sin = np.array([1,self.d,0,0])
        
    #Calculates the outcoming stokes vector for given derotator angle and half-waveplate angle
    def CalcSOut(self,ThetaDer,ThetaHwp):
        MCI = 0.5*Mt.ComMatrix(self.d,0)
        TDerMin = Mt.RotationMatrix(-(ThetaDer+self.DeltaDer))
        MDer = Mt.ComMatrix(self.eDer,self.RDer)
        TDerPlus = Mt.RotationMatrix(ThetaDer+self.DeltaDer)
        THwpMin = Mt.RotationMatrix(-(ThetaHwp+self.DeltaHwp))
        MHwp = Mt.ComMatrix(self.eHwp,self.RHwp)
        THwpPlus = Mt.RotationMatrix(ThetaHwp+self.DeltaHwp)
        TCal = Mt.RotationMatrix(-self.DeltaCal)

        return np.linalg.multi_dot([MCI,TDerMin,MDer,TDerPlus,THwpMin,MHwp,THwpPlus,TCal,self.Sin])
        
    def PlotEfficiencyDiagram(self,ThetaHwp,ThetaDerMin=0,ThetaDerMax=np.pi,N=400):
        
        PEfficiency = [] #Polarimetric efficiency
        PAngle = []
        ThetaDerList = np.linspace(ThetaDerMin,ThetaDerMax,N) 
        for ThetaDer in ThetaDerList:
            SOut = self.CalcSOut(ThetaDer,ThetaHwp)
            PEfficiency.append(Mt.LinPolDegree(SOut))
            PAngle.append(Mt.PolAngle(SOut)-Mt.PolAngle(self.Sin))
        
        ThetaDerList *= 180/np.pi
        ThetaDerMin *= 180/np.pi
        ThetaDerMax *= 180/np.pi        
        
        plt.plot(ThetaDerList,PEfficiency,label=self.Name)
        plt.xlim(xmin=ThetaDerMin,xmax=ThetaDerMax)
        plt.ylim(ymin=0,ymax=1.1)
        
        plt.xlabel("Derotator angle")
        plt.ylabel("Polarimetric efficiency")
        plt.xticks(np.arange(ThetaDerMin,ThetaDerMax,step=22.5))
        plt.yticks(np.arange(0,1.1,step=0.1))
                
          
        

#--/--FilterClass--/--#

#-----Parameters-----#

BB_Y = Filter("BB_Y",-0.00021,184.2,-0.6132,-0.00094,126.1,0.50007,0.9802,-1.542)
BB_J = Filter("BB_J",-0.000433,177.5,-0.6132,-0.008304,156.1,0.50007,0.9895,-1.542)
BB_H = Filter("BB_H",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542)
BB_K = Filter("BB_K",-0.000415,177.6,-0.6132,0.003552,84.13,0.50007,0.9842,-1.542)
#--/--Parameters-----#

#-----Main-----#

plt.title("Polarimetric efficiency vs Derotator angle")
BB_Y.PlotEfficiencyDiagram(0)
BB_J.PlotEfficiencyDiagram(0)
BB_H.PlotEfficiencyDiagram(0)
BB_K.PlotEfficiencyDiagram(0)
plt.legend()
plt.grid()  
plt.show()
#--/--Main-----#