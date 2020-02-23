import numpy as np
import matplotlib.pyplot as plt
import Methods as Mt

#-----FilterClass-----#

#The optical system of sphere for a specific wavelength
class MatrixModel():
    
    def __init__(self,Name,eHwp,RHwp,DeltaHwp,eDer,RDer,DeltaDer,d,DeltaCal,Steps1=100,Steps2=100):
        self.Name = Name #Name of filter
        self.eHwp = eHwp #Diattenuation of half-waveplate
        self.RHwp = RHwp *np.pi/180 #Retardance of half-waveplate
        self.DeltaHwp = DeltaHwp*np.pi/180 #Offset of half-waveplate
        self.eDer = eDer #Diattenuation of derotator
        self.RDer = RDer*np.pi/180 #Retardance of derotator
        self.DeltaDer = DeltaDer*np.pi/180 #Offset of derotator
        self.d = d #Diattenuation of polarizers
        self.DeltaCal = DeltaCal*np.pi/180 #Offset of calibration polarizer

        self.RunSetup1(Steps1)
        
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
          
        

#--/--FilterClass--/--#

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
        PEfficiency = np.sqrt(Model.Q1Out**2+Model.U1Out**2) / Model.I1Out
        
        plt.plot(ThetaDerList, PEfficiency, label=Model.Name)

    plt.legend()

#Plots Polarimetric efficiency vs derotator angle found using setup 1
def PlotOffsetDiagram(ModelList,Steps):
    
    ThetaDerList = np.linspace(0,180,Steps) #List of derotator angles

    plt.figure()

    plt.title("Offset angle vs Derotator angle")
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
            Sign=1
            if(Model.Q1Out[i] <= 0):
                Sign = -1
            
            POffset.append( Sign*0.5*np.arctan(Model.U1Out[i]/Model.Q1Out[i])*180/np.pi)
                       
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


#--/--Methods-----#

#-----Parameters-----#
Steps1 = 300
BB_Y = MatrixModel("BB_Y",-0.00021,184.2,-0.6132,-0.00094,126.1,0.50007,0.9802,-1.542,Steps1)
BB_J = MatrixModel("BB_J",-0.000433,177.5,-0.6132,-0.008304,156.1,0.50007,0.9895,-1.542,Steps1)
BB_H = MatrixModel("BB_H",-0.000297,170.7,-0.6132,-0.002260,99.32,0.50007,0.9955,-1.542,Steps1)
BB_K = MatrixModel("BB_K",-0.000415,177.6,-0.6132,0.003552,84.13,0.50007,0.9842,-1.542,Steps1)

ModelList = [BB_Y,BB_J,BB_H,BB_K]
PlotEfficiencyDiagram(ModelList,Steps1)
PlotOffsetDiagram(ModelList,Steps1)
PlotParameters1(BB_Y,Steps1)
plt.show()

#--/--Parameters-----#

#-----Main-----#


#--/--Main-----#