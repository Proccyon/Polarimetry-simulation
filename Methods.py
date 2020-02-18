#-----Header-----#
#This file contains methods for creating and applying
#matrices that represent optical elements.
#--/--Header--/--#


#-----Imports-----#
import numpy as np
#--/--Imports--/--#

#-----Matrices-----#

#Returns matrix of an ideal polarizer
#transmitting axis in +-Q direction
#Direction == True, --> axis in +Q direction
#Direction == False, --> axis in -Q direction
def CreatePolarizer(Direction):
    #DirectionConstant is either -1 or 1
    DirectionConstant = Direction*2-1
    PolarizerMatrix = np.zeros((4,4))
    PolarizerMatrix[0,0] = 1
    PolarizerMatrix[1,1] = 1
    PolarizerMatrix[0,1] = 1*DirectionConstant
    PolarizerMatrix[1,0] = 1*DirectionConstant
    return 0.5*PolarizerMatrix
   
#Returns matrix of an ideal retarder
#Fast axis aligned with +Q
#Delta is the retardance
def CreateRetarder(Delta):
    RetarderMatrix = np.zeros((4,4))
    RetarderMatrix[0,0] = 1
    RetarderMatrix[1,1] = 1
    RetarderMatrix[2,2] = np.cos(Delta)
    RetarderMatrix[3,2] = -np.sin(Delta)
    RetarderMatrix[2,3] = np.sin(Delta)
    RetarderMatrix[3,3] = np.cos(Delta)
    return RetarderMatrix
      
#Creates a RotationMatrix
#Use this to rotate optical elements
#Alpha is the angle of rotation from Q+ to U+
#Element is rotated by Mrot(-a)*M*Mrot(a)
def CreateRotationMatrix(Alpha):
    RotationMatrix = np.zeros((4,4)) 
    RotationMatrix[0,0] = 1
    RotationMatrix[3,3] = 1
    RotationMatrix[1,1] = np.cos(2*Alpha)
    RotationMatrix[2,1] = np.sin(-2*Alpha)
    RotationMatrix[1,2] = np.sin(2*Alpha)
    RotationMatrix[2,2] = np.cos(2*Alpha)
    return RotationMatrix
    
def IdentityMatrix():
    IdentityMatrix = np.zeros((4,4))
    IdentityMatrix[0,0] = 1
    IdentityMatrix[1,1] = 1
    IdentityMatrix[2,2] = 1
    IdentityMatrix[3,3] = 1
    return IdentityMatrix
    
#--/--Matrices--/--#

#-----Definitions-----#

#Calculates the degree of linear polarization
#S is a stokes vector
def PolarizationDegree(S):
    return np.sqrt(S[1]**2+S[2]**2+S[3]**2)/S[0]
    
#Calculates the linear polarization angle
def PolarizationAngle(S):
    return 0.5*np.arctan(S[2]/S[1])
    

#--/--Definitions--/--#

#-----OtherMethods-----#

#Measures Q using single difference method
#SField is a stokes vector as function of position(S = S())
def MeasureQ(SField):
    PositivePolarizer = CreatePolarizer(True)
    NegativePolarizer = CreatePolarizer(False)
    SFieldPlus = np.dot(PositivePolarizer,SField)
    SFieldMin = np.dot(NegativePolarizer,SField)
    return (SFieldPlus-SFieldMin)[0]                    
       
#Rotates an optical element by angle Alpha
def ApplyRotation(OpticalMatrix,Alpha):
    MatrixMin = CreateRotationMatrix(-Alpha)
    MatrixPlus = CreateRotationMatrix(Alpha)
    return np.dot(np.dot(MatrixMin,OpticalMatrix),MatrixPlus)                                                            
                                                                                                                                                                                                          
#--/--OtherMethods--/--#    

#-----FresnelEquations-----#

def FresnelRs(Na,Nb,Theta):
    Numerator = Na*np.cos(Theta)-np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    Denominator = Na*np.cos(Theta)+np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    return Numerator/Denominator

def FresnelRp(Na,Nb,Theta):
    Numerator = -np.cos(Theta)*Nb**2+Na*np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    Denominator = np.cos(Theta)*Nb**2+Na*np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    return Numerator/Denominator
    
def FresnelTs(Na,Nb,Theta):
    return 1-FresnelRs(Na,Nb,Theta)
    
def FresnelTp(Na,Nb,Theta):
    return 1-FresnelRp(Na,Nb,Theta)

#--/--FresnelEquations--/--#