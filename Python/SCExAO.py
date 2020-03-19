import matplotlib.pyplot as plt
import numpy as np
import Methods as Mt

#-----Classes-----#

#The optical system of sphere for a specific wavelength
class MatrixModel():
    
    def __init__(self,Name,E_Hwp,R_Hwp,DeltaHwp,E_Der,R_Der,DeltaDer,d,DeltaCal,E_UT,R_UT):
        self.Name = Name #Name of filter
        self.E_Hwp = E_Hwp #Diattenuation of half-waveplate
        self.R_Hwp = R_Hwp *np.pi/180 #Retardance of half-waveplate
        self.DeltaHwp = DeltaHwp*np.pi/180 #Offset of half-waveplate
        self.E_Der = E_Der #Diattenuation of derotator
        self.R_Der = R_Der*np.pi/180 #Retardance of derotator
        self.DeltaDer = DeltaDer*np.pi/180 #Offset of derotator
        self.d = d #Diattenuation of polarizers
        self.DeltaCal = DeltaCal*np.pi/180 #Offset of calibration polarizer
        
        self.E_UT = E_UT
        self.R_UT = R_UT *np.pi/180
        
    #Does intensity measurement with given system parameters
    def DoMeasurement(self,ThetaHwp,D_Sign,S_In,ThetaDer,Altitude,UsePolarizer=True):
        M_CI = 0.5*Mt.ComMatrix(self.d*D_Sign,0) #Polarizer for double difference
        
        T_DerMin = Mt.RotationMatrix(-(ThetaDer+self.DeltaDer)) #Derotator with rotation
        M_Der = Mt.ComMatrix(self.E_Der,self.R_Der)
        T_DerPlus = Mt.RotationMatrix(ThetaDer+self.DeltaDer)
        
        T_HwpMin = Mt.RotationMatrix(-(ThetaHwp+self.DeltaHwp)) #Half-waveplate with rotation
        M_Hwp = Mt.ComMatrix(self.E_Hwp,self.R_Hwp)
        T_HwpPlus = Mt.RotationMatrix(ThetaHwp+self.DeltaHwp)
        
        T_Cal = Mt.RotationMatrix(-self.DeltaCal)#Optional polarizer      
        M_Polarizer = Mt.ComMatrix(self.d,np.pi) 
        
        Ta = Mt.RotationMatrix(Altitude) #Telescope mirrors with rotation
        M_UT = Mt.ComMatrix(self.E_UT,self.R_UT)

        #TEST - REMOVE LATER
        M_M4 = Mt.ComMatrix(0.0182,171.9*np.pi/180)

        if(UsePolarizer):
            return np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,T_Cal,M_Polarizer,Ta,M_UT,S_In])[0]
        else:
            return np.linalg.multi_dot([M_CI,T_DerMin,M_Der,T_DerPlus,T_HwpMin,M_Hwp,T_HwpPlus,Ta,M_UT,S_In])[0]
        
        
    #Performs double difference with given system parameters
    def DoubleDifference(self,S_In,ThetaDer,Altitude,UsePolarizer):
        DoMeasurement = lambda ThetaHwp,D_Sign : self.DoMeasurement(ThetaHwp,D_Sign,S_In,ThetaDer,Altitude,UsePolarizer)
        
        #Find stokes Q
        XPlusQ = DoMeasurement(0,1) - DoMeasurement(0,-1) #Single difference for two hwp angles
        XMinQ = DoMeasurement(np.pi*(1/4),1) - DoMeasurement(np.pi*(1/4),-1)
        IPlusQ = DoMeasurement(0,1) + DoMeasurement(0,-1) #Single sum for two hwp angles
        IMinQ = DoMeasurement(np.pi*(1/4),1) + DoMeasurement(np.pi*(1/4),-1)
        Q = 0.5*(XPlusQ-XMinQ) #Double difference
        IQ = 0.5*(IPlusQ+IMinQ) #Double sum
        

        #Same thing but with hwp angles +(1/8)pi to get stokes U
        XPlusU = DoMeasurement((1/8)*np.pi,1) - DoMeasurement((1/8)*np.pi,-1)
        XMinU = DoMeasurement((3/8)*np.pi,1) - DoMeasurement((3/8)*np.pi,-1)
        IPlusU = DoMeasurement((1/8)*np.pi,1) + DoMeasurement((1/8)*np.pi,-1)
        IMinU = DoMeasurement((3/8)*np.pi,1) + DoMeasurement((3/8)*np.pi,-1)
        U = 0.5*(XPlusU-XMinU)
        IU = 0.5*(IPlusU+IMinU)

        I = 0.5*(IQ+IU) #IQ should be the same as IU but we average anyway
        return I,Q,U #Returns parameters seperately as we do not know stokes V

    
    #Measures stokes vector for different derotator angles
    def RunPE_Measurement(self,Steps):
        IOut = []
        QOut = []
        UOut = []
        ThetaDerList = np.linspace(0,np.pi,Steps) #List of derotator angles

        for ThetaDer in ThetaDerList:
            I,Q,U = self.DoubleDifference(np.array([1,0,0,0]),ThetaDer,0,True)
            IOut.append(I)
            QOut.append(Q)
            UOut.append(U)

        return np.array(IOut),np.array(QOut),np.array(UOut)

        #Measures stokes vector for different derotator angles
    def RunIP_Measurement(self,Steps):
        IOut = []
        QOut = []
        UOut = []
        AltitudeList = np.linspace(0,0.5*np.pi,Steps) #List of derotator angles

        for Altitude in AltitudeList:
            I,Q,U = self.DoubleDifference(np.array([1,0,0,0]),0,Altitude,False)
            IOut.append(I)
            QOut.append(Q)
            UOut.append(U)

        return np.array(IOut),np.array(QOut),np.array(UOut)

#--/--Classes--/--#

#-----Methods-----#
#Plots Polarimetric efficiency vs derotator angle found using setup 1
def PlotEfficiencyDiagram(ModelList,Steps):
    
    ThetaDerList = np.linspace(0,180,Steps) #List of derotator angles

    plt.figure()

    plt.title("Polarimetric efficiency vs Derotator angle")
    plt.xlabel("Derotator angle")
    plt.ylabel("Polarimetric efficiency")

    plt.xlim(xmin=0,xmax=180)
    plt.ylim(ymin=0,ymax=1.1)
    plt.xticks(np.arange(0,180,step=22.5))
    plt.yticks(np.arange(0,1.1,step=0.1))   
    
    plt.grid() 

    for Model in ModelList:
        IOut,QOut,UOut = Model.RunPE_Measurement(Steps)
        PE = np.sqrt(QOut**2 + UOut**2) / IOut      
        plt.plot(ThetaDerList,PE, label=Model.Name)

    plt.legend()

def PlotIpDiagram(ModelList,Steps):
    
    AltitudeList = np.linspace(0,90,Steps) #List of derotator angles

    plt.figure()
    plt.title("Altitude vs IP")
    plt.xlabel("Altitude")
    plt.ylabel("Instrumental polarization(%)")

    plt.xlim(xmin=0,xmax=90)
    plt.ylim(ymin=0,ymax=5)
    plt.xticks(np.arange(0,90,step=10))
    plt.yticks(np.arange(0,5,step=0.5))   
    
    plt.grid() 

    for Model in ModelList:
        IOut,QOut,UOut = Model.RunIP_Measurement(Steps)
        IP = 100*np.sqrt(QOut**2 + UOut**2) / IOut     
        plt.plot(AltitudeList,IP, label=Model.Name)

    plt.legend()
    
#--/--Methods--/--#




#-----Main-----#
Steps = 100
BB_Y = MatrixModel("BB_Y",-0.00021,184.2,-0.6132,-0.00094,126.1,0.50007,0.9802,-1.542,0.0236,171.9)
BB_J = MatrixModel("BB_J",-0.000433,177.5,-0.6132,-0.008304,156.1,0.50007,0.9895,-1.542,0.0167,173.4)
BB_H = MatrixModel("BB_H",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542,0.01293,175)
BB_K = MatrixModel("BB_K",-0.000415,177.6,-0.6132,0.003552,84.13,0.50007,0.9842,-1.542,0.0106,176.3)

ModelList = [BB_Y,BB_J,BB_H,BB_K]
PlotEfficiencyDiagram(ModelList,Steps)
PlotIpDiagram(ModelList,Steps)
plt.show()
#--/--Main--/--#
         