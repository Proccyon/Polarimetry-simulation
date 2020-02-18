#-----Header-----#
#This file tests the methods in Methods.py by applying matrices
#to a generated observation.
#The observation consists of a star emitting unpolarized light 
#and a planet reflecting Q+ polarized light
#In the first test the Q component of the stokes vector 
#is measured using the single difference method.
#In the second test IP is introduced and corrected using 
#the double difference method.
#
#--/--header--/--#

#-----Imports-----#
import numpy as np
import matplotlib.pyplot as plt
import Methods as Mt
#--/--Imports--/--#

#-----Parameters-----#

Width = 100 #Width(and height) of field in pixels
StarI0 = 30
StarR0 = 10
PlanetX = 10 #Position of planet compared to the middle
PlanetI0 = 3
PlanetR0 = 2
#--/--Parameters--/--#


#-----Methods-----#
#Calculates intensity of star as function of position
#Star intensity falls off exponentially 
#Star is placed at (0,0)
def StarIntensity(X,Y,I0,R0):
    R = np.sqrt(X**2+Y**2)
    return I0*np.exp(-R/R0)

#Calculates intensity of planet as function of position
#Planet intensity falls off exponentially
#Planet is placed at (dx,0)
#In this case Q = I
def PlanetIntensity(X,Y,I0,R0,dx):
    R = np.sqrt((X-dx)**2+Y**2) 
    return I0*np.exp(-R/R0)
    
#Simulates the observation field
#Returns Stokes field
def SimulateObservation(Width,StarI0,StarR0,PlanetI0,PlanetR0,PlanetX):
    SField = np.empty((4),dtype=np.ndarray) #Creates empty stokes field
    SField[0] = np.zeros((Width,Width))
    SField[1] = np.zeros((Width,Width))
    SField[2] = np.zeros((Width,Width))
    SField[3] = np.zeros((Width,Width))
    for x in range(Width):
        for y in range(Width):
            X = x-Width/2 #Position relative to the middle
            Y = y-Width/2
            #Sets the intensity of the field
            SField[0][x,y] = StarIntensity(X,Y,StarI0,StarR0)+PlanetIntensity(X,Y,PlanetI0,PlanetR0,PlanetX)
            #Sets the Q parameter of the field
            SField[1][x,y] = PlanetIntensity(X,Y,PlanetI0,PlanetR0,PlanetX)
    
    return SField
    
#--/--Methods--/--#

#-----Main-----#
SField = SimulateObservation(Width,StarI0,StarR0,PlanetI0,PlanetR0,PlanetX)

#-----ShowIntensity-----#
#Shows the intensity of the Stokes field
if(False):
    plt.figure()
    plt.title("Total intensity")
    plt.imshow(SField[0])
    plt.colorbar()
#--/--ShowIntensity-----#

#-----SingleDifference-----#
#Calculates Q using single difference method
if(True):    
    QDifference = Mt.MeasureQ(SField) #This uses single difference method

    plt.figure() #Q found directly from stokes vector.
    plt.title("Actual Q image")
    plt.imshow(SField[1])
    plt.colorbar()
    
    plt.figure() #Q found using single difference method, it's the same thing.
    plt.title("Q measured using double difference method")
    plt.imshow(QDifference)
    plt.colorbar()

#--/--SingleDifference--/--#

#-----DoubleDifference-----#
#Measures Q again but with double difference method
#Now Ip is introduced
if(False):
    Ip = 0.1 #Amount of IP introduced, Ip < 1
    IpMatrix = Mt.IdentityMatrix() #The matrix that introduces IP
    IpMatrix[1,0] = Ip #Technically I should decrease so IpMatrix[0,0] < 1
    
    #Matrix that turns +Q into -Q
    RetarderMatrix = Mt.ApplyRotation(Mt.CreateRetarder(np.pi),0.25*np.pi)
       
    SFieldRot = np.dot(RetarderMatrix,SField)#Field rotated by +Pi
    SFieldIp = np.dot(IpMatrix,SField) #Field with IP
    SFieldRotIp = np.dot(IpMatrix,SFieldRot) #Rotated field with IP
    QCorrected = (Mt.MeasureQ(SFieldIp)-Mt.MeasureQ(SFieldRotIp))/2 #QField corrected for IP
    
    plt.figure() #Shows Q field if we had not corrected for IP(Single difference)
    plt.title("Q not corrected for Ip")
    plt.imshow(Mt.MeasureQ(SFieldIp))
    plt.colorbar()
    
    plt.figure() #Shows Q field corrected for IP
    plt.title("Q corrected for Ip")
    plt.imshow(QCorrected)
    plt.colorbar()

#--/--DoubleDifference--/--#

plt.show()
#--/--Main--/--#