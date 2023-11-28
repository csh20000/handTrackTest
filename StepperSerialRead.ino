#include "pitches.h"
#define MAX_MILLIS_TO_WAIT 1000  //or whatever
#define MAXBYTES 10
///#include <Stepper.h>

unsigned long starttime;

// Define pin connections & motor's steps per revolution
const int dirPin_M1 = 5;
const int stepPin_M1 = 2;

const int dirPin_M2 = 6;
const int stepPin_M2 = 3;

const int dirPin_M3 = 7;
const int stepPin_M3 = 4;

const int stepPin_M4 = 12;

const int stepsPerRevolution = 200;
int pitchVal[10];
int i = 0;
byte bytes[10];
String ReceivedString[10];
char midMarker = 'n';
char endMarker = 'A';
String input;

char receivedNote[MAXBYTES];
bool newData = false;
int iCount = 0;
int SuperMario =0;

bool pedalPressed = 0;
int SustainPedal[8];
int pedalIndex= 0;

unsigned long motorSpeeds[] = {0,0,0}; //holds the speeds of the motors. 
unsigned long prevStepMicros[] = {0,0,0}; //last time
bool disableSteppers = HIGH; //status of the enable pin. disabled when HIGH. Gets enabled when the first note on message is received.
unsigned long WDT; //Will store the time that the last event occured.
unsigned long timer = 0;
#define TIMEOUT 10000 //Number of milliseconds for watchdog timer


int StringSplit(String sInput, char cDelim, String sParams[], int iMaxParams);


String getValue(String data, char separator, int index)
{
  int found = 0;
  int strIndex[] = {0, -1};
  int maxIndex = data.length()-1;

  for(int i=0; i<=maxIndex && found<=index; i++){
    if(data.charAt(i)==separator || i==maxIndex){
        found++;
        strIndex[0] = strIndex[1]+1;
        strIndex[1] = (i == maxIndex) ? i+1 : i;
    }
  }

  return found>index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void handleNoteOn(byte channel, byte pitch, byte velocity); //MIDI Note ON Command


void handleNoteOff(byte channel, byte pitch, byte velocity); //MIDI Note OFF Command


void singleStep(byte motorNum, byte stepPin);


void setup()
{
  Serial.begin(38400);
  Serial.setTimeout(3);
  // Declare pins as Outputs
  pinMode(stepPin_M1, OUTPUT);
  pinMode(dirPin_M1, OUTPUT);
  pinMode(stepPin_M2, OUTPUT);
  pinMode(dirPin_M2, OUTPUT);
  pinMode(stepPin_M3, OUTPUT);
  pinMode(dirPin_M3, OUTPUT);
  pinMode(A0,INPUT);
}
void loop()
{
  // Set motor direction clockwise
  digitalWrite(dirPin_M1, HIGH);
  digitalWrite(dirPin_M2, HIGH);
  digitalWrite(dirPin_M3, HIGH);
  if(SuperMario==0)
  {
    int Last_State[10];
    starttime = millis();

    SustainPedal[pedalIndex%4] = analogRead(A0);
    if(SustainPedal[0]==0 && SustainPedal[1] == 0 && SustainPedal[2]==0 && SustainPedal[3] == 0 && SustainPedal[4]==0 && SustainPedal[5] == 0 && SustainPedal[6]==0 && SustainPedal[7] == 0)
    {
      pedalPressed = 0;
    }
    else
    {
      pedalPressed = 1;
    }
    //recWithEndMark();
    while(Serial.available()!=0 && ((millis() - starttime) < MAX_MILLIS_TO_WAIT) )
    {
      String last_state = input;
       //Serial.println("serial available");
      input = Serial.readStringUntil('A');
  
  
      //Serial.print("Pitch Value: ");
      //Serial.print("[");
      for(int i= 0; i<10; i++)
      { 
      
        Last_State[i] = getValue(last_state,'n',i).toInt();
        if(Last_State[i] == getValue(input,'n',i).toInt())
        {
          //if(pedalPressed==0)
          //{ 
            pitchVal[i]=0;
          //}
           //else if(pedalPressed == 1)
          //{
           ///pitchVal[i] = Last_State[i];
          //}
        }
        else
        {
          pitchVal[i] = getValue(input,'n',i).toInt();
        }
        //Serial.print(pitchVal[i]);
        //Serial.print(",");
      }
      String pitchConc;
      String LastVal_Conc;
      pitchConc.concat(pitchVal[0]); 
      pitchConc.concat(pitchVal[1]); 
      pitchConc.concat(pitchVal[2]); 
      pitchConc.concat(pitchVal[3]); 
      pitchConc.concat(pitchVal[4]); 
      pitchConc.concat(pitchVal[5]); 
      pitchConc.concat(pitchVal[6]); 
      pitchConc.concat(pitchVal[7]); 
      pitchConc.concat(pitchVal[8]); 
      pitchConc.concat(pitchVal[9]);   

      LastVal_Conc.concat(Last_State[0]);
      LastVal_Conc.concat(Last_State[1]);
      LastVal_Conc.concat(Last_State[2]);
      LastVal_Conc.concat(Last_State[3]);
      LastVal_Conc.concat(Last_State[4]);
      LastVal_Conc.concat(Last_State[5]);
      LastVal_Conc.concat(Last_State[6]);
      LastVal_Conc.concat(Last_State[7]);
      LastVal_Conc.concat(Last_State[8]);
      LastVal_Conc.concat(Last_State[9]);
      
      //Serial.print("]");
      //Serial.println("");
      Serial.print("Data Received: ");
      Serial.print(input);
      Serial.print("  Last State: ");
      Serial.print(last_state);
      Serial.print("  Last State Conc: ");
      Serial.print(LastVal_Conc);
      Serial.print("  Pitch Value: ");
      Serial.print(pitchConc);
      Serial.println("");
    }
    int Fifo_Motor_num =0;
    if(getValue(input,'n',5)==0 && getValue(input,'n',6)==0 && getValue(input,'n',7)==0 && getValue(input,'n',8)==0 && getValue(input,'n',9)==0)
    {
      Fifo_Motor_num = 5;
    }
    else
    {
      Fifo_Motor_num = 10;
    }
    for(int i=0; i<10; i++)
    {
      if(pitchVal[i]<72 || pitchVal[i]>100)
      {
        pitchVal[i]=0;
      }
      // Spin motor slowly
      else if(pitchVal[i]>71 && pitchVal[i]<101)
      {
        if(i==10)
        {
          break;
        }
        while(pitchVal[i]==0)
        {
          i++;
        }
        
        if(i%3==0)
        {
          for(int x = 0; x < 5*stepsPerRevolution/(127-int(pitchVal[i])); x++)
          {
            //Motor 1
            digitalWrite(stepPin_M1, HIGH);
            delayMicroseconds(pitchVals[pitchVal[i]]);
            digitalWrite(stepPin_M1, LOW);
            delayMicroseconds(pitchVals[pitchVal[i]]);
          }
        }
        if(i%3==1)
        {
          for(int x = 0; x < 5*stepsPerRevolution/(127-int(pitchVal[i])); x++)
          {
            //Motor 2
            digitalWrite(stepPin_M2, HIGH);
            delayMicroseconds(pitchVals[pitchVal[i]]);
            digitalWrite(stepPin_M2, LOW);
            delayMicroseconds(pitchVals[pitchVal[i]]);
          }
        }
        if(i%3==2)
        {
          for(int x = 0; x < 5*stepsPerRevolution/(127-int(pitchVal[i])); x++)
          {
            //Motor 3
            digitalWrite(stepPin_M3, HIGH);
            delayMicroseconds(pitchVals[pitchVal[i]]);
            digitalWrite(stepPin_M3, LOW);
            delayMicroseconds(pitchVals[pitchVal[i]]);
          }
        }
      
        else
        {
          digitalWrite(stepPin_M1, LOW);
          digitalWrite(stepPin_M2, LOW);
          digitalWrite(stepPin_M3, LOW);
          digitalWrite(stepPin_M4, LOW);
        }
      }
      pedalIndex = pedalIndex+1;
      pitchVal[i]=0;
    }
    newData =false;

    /*
    singleStep(0, stepPin_M1);
    singleStep(1, stepPin_M2);
    singleStep(2, stepPin_M3);
  
    for(int i=0; i<5; i++)
    {
      
      if(pitchVal[i] > 22 && motorSpeeds[i%3] != pitchVals[pitchVal[i]])
      {
        handleNoteOn(i,pitchVal[i],0);     
      }
      else if(pitchVal[i]<23 && motorSpeeds[i%3] != 0)
      {
        handleNoteOff(i,pitchVal[i],0);
      }
    }*/
  }
}

void handleNoteOn(byte channel, byte pitch, byte velocity) //MIDI Note ON Command
{
  disableSteppers = LOW; //enable steppers. 
  motorSpeeds[channel] = pitchVals[pitch]; //set the motor speed to specified pitch
}

void handleNoteOff(byte channel, byte pitch, byte velocity) //MIDI Note OFF Command
{
  motorSpeeds[channel] = 0; //set motor speed to zero
}

void singleStep(byte motorNum, byte stepPin)
{
  if ((micros() - prevStepMicros[motorNum] >= motorSpeeds[motorNum]) && (motorSpeeds[motorNum] != 0)) 
  { //step when correct time has passed and the motor is at a nonzero speed
    prevStepMicros[motorNum] += motorSpeeds[motorNum];
    WDT = millis(); //update watchdog timer
    digitalWrite(stepPin, HIGH);
    digitalWrite(stepPin, LOW);
  }
}


int StringSplit(String sInput, char cDelim, String sParams[], int iMaxParams)
{
    int iParamCount = 0;
    int iPosDelim, iPosStart = 0;

    do {
        // Searching the delimiter using indexOf()
        iPosDelim = sInput.indexOf(cDelim,iPosStart);
        if (iPosDelim > (iPosStart+1)) {
            // Adding a new parameter using substring() 
            sParams[iParamCount] = sInput.substring(iPosStart,iPosDelim-1);
            iParamCount++;
            // Checking the number of parameters
            if (iParamCount >= iMaxParams) {
                return (iParamCount);
            }
            iPosStart = iPosDelim + 1;
        }
    } while (iPosDelim >= 0);
    if (iParamCount < iMaxParams) {
        // Adding the last parameter as the end of the line
        sParams[iParamCount] = sInput.substring(iPosStart);
        iParamCount++;
    }

    return (iParamCount);
}
