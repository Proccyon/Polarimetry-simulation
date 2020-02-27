import numpy as np
import matplotlib.pyplot as plt
import Methods as Mt

#-----Header-----#

#This file reproduces some of the results found in Holstein et al.
#It reproduces figures 5,6,8 and 9 by simulating the optical elements
#Using mueller matrices.
#Setup 1 tests the optical elements downstream of the half-waveplate by introducing 
#fully Q+ polarized light.
#Setup 2 tests the optical elements upstream of the half-waveplate by introducing 
#fully unpolarized light.

#--/--Header--/--#


#-----FilterClass-----#

#The optical system of sphere for a specific wavelength
class MatrixModel():
    
    def __init__(self,Name,eHwp,RHwp,DeltaHwp,eDer,RDer,DeltaDer,d,DeltaCal,eUT,eM4,RUT,RM4,Steps=100):
        self.Name = Name #Name of filter
        self.eHwp = eHwp #Diattenuation of half-waveplate
        self.RHwp = RHwp *np.pi/180 #Retardance of half-waveplate
        self.DeltaHwp = DeltaHwp*np.pi/180 #Offset of half-waveplate
        self.eDer = eDer #Diattenuation of derotator
        self.RDer = RDer*np.pi/180 #Retardance of derotator
        self.DeltaDer = DeltaDer*np.pi/180 #Offset of derotator
        self.d = d #Diattenuation of polarizers
        self.DeltaCal = DeltaCal*np.pi/180 #Offset of calibration polarizer
        
        self.eUT = eUT
        self.eM4 = eM4
        self.RUT = RUT *np.pi/180
        self.RM4 = RM4 *np.pi/180

        self.RunSetup1(Steps)
        self.RunSetup2(Steps)
        
    #-----Setup1-----#

    #Does intensity measurement for setup 1 at given derotator angle, half-waveplate angle and polarizer sign
    def DoMeasurement1(self,ThetaDer,ThetaHwp,DSign):
        MCI = 0.5*Mt.ComMatrix(self.d*DSign,0)
        TDerMin = Mt.RotationMatrix(-(ThetaDer+self.DeltaDer))
        MDer = Mt.ComMatrix(self.eDer,self.RDer)
        TDerPlus = Mt.RotationMatrix(ThetaDer+self.DeltaDer)
        THwpMin = Mt.RotationMatrix(-(ThetaHwp+self.DeltaHwp))
        MHwp = Mt.ComMatrix(self.eHwp,self.RHwp)
        THwpPlus = Mt.RotationMatrix(ThetaHwp+self.DeltaHwp)
        TCal = Mt.RotationMatrix(-self.DeltaCal)
        S1In = np.array([1,self.d,0,0])

        return np.linalg.multi_dot([MCI,TDerMin,MDer,TDerPlus,THwpMin,MHwp,THwpPlus,TCal,S1In])[0]
        
    #Finds outcoming stokes parameters for setup 1 by applying single and dubble difference method. Does not return stokes V.
    def GetStokesOut1(self,ThetaDer):
        #Find stokes Q
        XPlusQ = self.DoMeasurement1(ThetaDer,0,1) - self.DoMeasurement1(ThetaDer,0,-1) #Single difference for two hwp angles
        XMinQ = self.DoMeasurement1(ThetaDer,np.pi*(1/4),1) - self.DoMeasurement1(ThetaDer,np.pi*(1/4),-1)
        IPlusQ = self.DoMeasurement1(ThetaDer,0,1) + self.DoMeasurement1(ThetaDer,0,-1) #Single sum for two hwp angles
        IMinQ = self.DoMeasurement1(ThetaDer,np.pi*(1/4),1) + self.DoMeasurement1(ThetaDer,np.pi*(1/4),-1)
        Q = 0.5*(XPlusQ-XMinQ) #Double difference
        IQ = 0.5*(IPlusQ+IMinQ) #Double sum

        #Same thing but with hwp angles +(1/8)pi to get stokes U
        XPlusU = self.DoMeasurement1(ThetaDer,(1/8)*np.pi,1) - self.DoMeasurement1(ThetaDer,(1/8)*np.pi,-1)
        XMinU = self.DoMeasurement1(ThetaDer,(1/8)*np.pi+np.pi*(1/4),1) - self.DoMeasurement1(ThetaDer,(3/8)*np.pi,-1)
        IPlusU = self.DoMeasurement1(ThetaDer,(1/8)*np.pi,1) + self.DoMeasurement1(ThetaDer,(1/8)*np.pi,-1)
        IMinU = self.DoMeasurement1(ThetaDer,(1/8)*np.pi+np.pi*(1/4),1) + self.DoMeasurement1(ThetaDer,(3/8)*np.pi,-1)
        U = 0.5*(XPlusU-XMinU)
        IU = 0.5*(IPlusU+IMinU)
        
        I = 0.5*(IQ+IU) #IQ should be the same as IU but we average anyway
        return I,Q,U #Returns parameters seperately as we do not know stokes V

    #Runs GetStokesOut1 for Derotator angles from 0 to 180 degrees
    def RunSetup1(self,Steps):
        I1Out = []
        Q1Out = []
        U1Out = []
        ThetaDerList = np.linspace(0,np.pi,Steps) #List of derotator angles

        for ThetaDer in ThetaDerList:
            I,Q,U = self.GetStokesOut1(ThetaDer)
            I1Out.append(I)
            Q1Out.append(Q)
            U1Out.append(U)

        self.I1Out = np.array(I1Out)
        self.Q1Out = np.array(Q1Out)
        self.U1Out = np.array(U1Out)

    #--/--Setup1--/--#        
          
    #-----Setup2-----#
    
    def DoMeasurement2(self,Altitude,ThetaHwp,DSign):
        MCI = 0.5*Mt.ComMatrix(self.d*DSign,0)
        MDer = Mt.ComMatrix(self.eDer,self.RDer)
        THwpMin = Mt.RotationMatrix(-(ThetaHwp+self.DeltaHwp))
        MHwp = Mt.ComMatrix(self.eHwp,self.RHwp)
        THwpPlus = Mt.RotationMatrix(ThetaHwp+self.DeltaHwp)
        MM4 = Mt.ComMatrix(self.eM4,self.RM4)
        Ta = Mt.RotationMatrix(Altitude)
        MUT = Mt.ComMatrix(self.eUT,self.RUT)
        S2In = np.array([1,0,0,0])

        return np.linalg.multi_dot([MCI,MDer,THwpMin,MHwp,THwpPlus,MM4,Ta,MUT,S2In])[0]
    
    
    #Finds outcoming stokes parameters for setup 2 by applying single and dubble difference method. Does not return stokes V.
    def GetStokesOut2(self,Altitude):
        #Find stokes Q
        XPlusQ = self.DoMeasurement2(Altitude,0,1) - self.DoMeasurement2(Altitude,0,-1) #Single difference for two hwp angles
        XMinQ = self.DoMeasurement2(Altitude,np.pi*(1/4),1) - self.DoMeasurement2(Altitude,np.pi*(1/4),-1)
        IPlusQ = self.DoMeasurement2(Altitude,0,1) + self.DoMeasurement2(Altitude,0,-1) #Single sum for two hwp angles
        IMinQ = self.DoMeasurement2(Altitude,np.pi*(1/4),1) + self.DoMeasurement2(Altitude,np.pi*(1/4),-1)
        Q = 0.5*(XPlusQ-XMinQ) #Double difference
        IQ = 0.5*(IPlusQ+IMinQ) #Double sum

        #Same thing but with hwp angles +(1/8)pi to get stokes U
        XPlusU = self.DoMeasurement2(Altitude,(1/8)*np.pi,1) - self.DoMeasurement2(Altitude,(1/8)*np.pi,-1)
        XMinU = self.DoMeasurement2(Altitude,(1/8)*np.pi+np.pi*(1/4),1) - self.DoMeasurement2(Altitude,(3/8)*np.pi,-1)
        IPlusU = self.DoMeasurement2(Altitude,(1/8)*np.pi,1) + self.DoMeasurement2(Altitude,(1/8)*np.pi,-1)
        IMinU = self.DoMeasurement2(Altitude,(1/8)*np.pi+np.pi*(1/4),1) + self.DoMeasurement2(Altitude,(3/8)*np.pi,-1)
        U = 0.5*(XPlusU-XMinU)
        IU = 0.5*(IPlusU+IMinU)
        
        I = 0.5*(IQ+IU) #IQ should be the same as IU but we average anyway
        return I,Q,U #Returns parameters seperately as we do not know stokes V
    
    #Runs GetStokesOut2 for Altitudes from 0 to 90 degrees
    def RunSetup2(self,Steps):
        I2Out = []
        Q2Out = []
        U2Out = []
        AltitudeList = np.linspace(0,0.5*np.pi,Steps) #List of derotator angles

        for Altitude in AltitudeList:
            I,Q,U = self.GetStokesOut2(Altitude)
            I2Out.append(I)
            Q2Out.append(Q)
            U2Out.append(U)

        self.I2Out = np.array(I2Out)
        self.Q2Out = np.array(Q2Out)
        self.U2Out = np.array(U2Out)
    #--/--Setup2--/--#

#--/--FilterClass--/--#

#-----Methods-----#

def Modulate(Angle):
    while(Angle <-90 or Angle > 90):
        if(Angle < -90):
            Angle += 180
        if(Angle > 90):
            Angle -= 180
    return Angle

#Plots Polarimetric efficiency vs derotator angle found using setup 1
def PlotEfficiencyDiagram(ModelList,Steps):
    
    ThetaDerList = np.linspace(0,180,Steps) #List of derotator angles

    plt.figure()

    plt.title("Polarimetric efficiency vs Derotator angle (f5)")
    plt.xlabel("Derotator angle")
    plt.ylabel("Polarimetric efficiency")

    plt.xlim(xmin=0,xmax=180)
    plt.ylim(ymin=0,ymax=1.1)
    plt.xticks(np.arange(0,180,step=22.5))
    plt.yticks(np.arange(0,1.1,step=0.1))   
    
    plt.grid() 

    for Model in ModelList:
        PEfficiency = np.sqrt(Model.Q1Out**2+Model.U1Out**2) / Model.I1Out       
        plt.plot(ThetaDerList, PEfficiency, label=Model.Name)

    plt.legend()

#Plots Polarimetric efficiency vs derotator angle found using setup 1
def PlotOffsetDiagram(ModelList,Steps):
    
    ThetaDerList = np.linspace(0,180,Steps) #List of derotator angles

    plt.figure()

    plt.title("Offset angle vs Derotator angle (f6)")
    plt.xlabel("Derotator angle")
    plt.ylabel("Offset of angle of linear polarization")

    plt.xlim(xmin=0,xmax=180)
    plt.ylim(ymin=-90,ymax=90)
    plt.xticks(np.arange(0,180,step=22.5))
    plt.yticks(np.arange(-90,90,step=15))   
    
    plt.grid() 

    for Model in ModelList:
        POffset = []
        for i in range(len(Model.U1Out)):
            Delta = 0
            if(Model.Q1Out[i] <= 0):
                Delta = 90
            
            NewOffset = 0.5*np.arctan(Model.U1Out[i]/Model.Q1Out[i])*180/np.pi+Delta - 2*ThetaDerList[i] 
            NewOffset = Modulate(NewOffset)
            POffset.append( NewOffset)
        
                       
        POffset = np.array(POffset)                                            
        plt.plot(ThetaDerList, POffset, label=Model.Name)

    plt.legend()

#Plots Polarimetric efficiency vs derotator angle found using setup 1
def PlotParameters1(Model,Steps):
    
    ThetaDerList = np.linspace(0,180,Steps) #List of derotator angles

    plt.figure()

    plt.title("Q and U vs derotator angle for "+Model.Name)
    plt.xlabel("Derotator angle")
    plt.ylabel("Value of Q or U")

    plt.xlim(xmin=0,xmax=180)
    plt.ylim(ymin=-2,ymax=2)

    plt.xticks(np.arange(0,180,step=22.5))
    
    plt.grid() 

    plt.plot(ThetaDerList,Model.Q1Out,label="Q")
    plt.plot(ThetaDerList,Model.U1Out,label="U")
    plt.plot(ThetaDerList,Model.U1Out/Model.Q1Out,label="U/Q")

    plt.legend()

#Plots instrumentel polarization vs altitude
def PlotInstrumentalPolarization(ModelList,Steps,Title=""):
    
    AltitudeList = np.linspace(0,90,Steps)
    
    plt.figure()

    plt.title(Title)
    plt.xlabel("Altitude")
    plt.ylabel("Instrumental polarization")

    plt.xlim(xmin=0,xmax=90)
    plt.ylim(ymin=0,ymax=4.5)
    plt.xticks(np.arange(0,90,step=10))
    plt.yticks(np.arange(0,4.5,step=0.5))   
    
    plt.grid() 

    for Model in ModelList:
        IP = (np.sqrt(Model.Q2Out**2+Model.U2Out**2) / Model.I2Out ) * 100                                               
        plt.plot(AltitudeList, IP, label=Model.Name)

    plt.legend()

#--/--Methods-----#

#-----Parameters-----#
Steps = 300
BB_Y_b = MatrixModel("BB_Y_b",-0.00021,184.2,-0.6132,-0.00094,126.1,0.50007,0.9802,-1.542,0.0236,0.0182,171.9,171.9,Steps)
BB_J_b = MatrixModel("BB_J_b",-0.000433,177.5,-0.6132,-0.008304,156.1,0.50007,0.9895,-1.542,0.0167,0.0128,173.4,173.4,Steps)
BB_H_b = MatrixModel("BB_H_b",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542,0.01293,0.00985,175,175,Steps)
BB_K_b = MatrixModel("BB_K_b",-0.000415,177.6,-0.6132,0.003552,84.13,0.50007,0.9842,-1.542,0.0106,0.0078,176.3,176.3,Steps)

BB_Y_a = MatrixModel("BB_Y_a",-0.00021,184.2,-0.6132,-0.00094,126.1,0.50007,0.9802,-1.542,0.0175,0.0182,171.9,171.9,Steps)
BB_J_a = MatrixModel("BB_J_a",-0.000433,177.5,-0.6132,-0.008304,156.1,0.50007,0.9895,-1.542,0.0121,0.0130,173.4,173.4,Steps)
BB_H_a = MatrixModel("BB_H_a",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542,0.0090,0.0092,175,175,Steps)
BB_K_a = MatrixModel("BB_K_a",-0.000415,177.6,-0.6132,0.003552,84.13,0.50007,0.9842,-1.542,0.0075,0.0081,176.3,176.3,Steps)
ModelListB = [BB_Y_b,BB_J_b,BB_H_b,BB_K_b]
ModelListA = [BB_Y_a,BB_J_a,BB_H_a,BB_K_a]

PlotEfficiencyDiagram(ModelListB,Steps)
PlotOffsetDiagram(ModelListB,Steps)

PlotInstrumentalPolarization(ModelListB,Steps,"Instrumental polarization before re-aluminizatio (f8)")
PlotInstrumentalPolarization(ModelListA,Steps,"Instrumental polarization after re-aluminizatio (f9)")

plt.show()

#--/--Parameters-----#

#-----Main-----#


#--/--Main-----#