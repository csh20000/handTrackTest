#include "pitches.h"
#define MAX_MILLIS_TO_WAIT 1000  //or whatever

unsigned long starttime;

// Define pin connections & motor's steps per revolution
const int dirPin_M1 = 5;
const int stepPin_M1 = 2;

const int dirPin_M2 = 6;
const int stepPin_M2 = 3;

const int dirPin_M3 = 7;
const int stepPin_M3 = 4;

const int stepsPerRevolution = 200;
int i = 0;

int pitchVal[10];
char midMarker = 'n';
char endMarker = 'A';
String input;

int iCount = 0;

bool pedalPressed = 0;
int SustainPedal[8];
int pedalIndex= 0;
int stepSize = 0;
float counter;


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
  
  int Last_State[10];
  starttime = millis();

  SustainPedal[pedalIndex%4] = analogRead(A0);

  pedalPressed = SustainPedal[0]==0 && SustainPedal[1]==0 && SustainPedal[2]==0 && SustainPedal[3]==0 && SustainPedal[4]==0 && SustainPedal[5]==0 && SustainPedal[6]==0 && SustainPedal[7]==0 ? 1 : 0;
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
        if(pedalPressed==0)
        { 
          pitchVal[i]=0;
        }
         else if(pedalPressed == 1)
        {
         pitchVal[i] = Last_State[i];
        }
      }
      else
      {
        pitchVal[i] = getValue(input,'n',i).toInt();
      }
      //Serial.print(pitchVal[i]);
      //Serial.print(",");

      if(pitchVal[i] != 0)
      {
        counter++;
      }
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


  for(int i=0; i<10; i++)
  {
  
    stepSize = 8*stepsPerRevolution/(counter*(127-int(pitchVal[i])));
    if(pitchVal[i]<72 || pitchVal[i]>100)
    {
      pitchVal[i]=0;
    }
    // Spin motor slowly
    else if(pitchVal[i]>71 && pitchVal[i]<101)
    {
      while(pitchVal[i]==0)
      {
        i++;
      }
      for(int x = 0; x < stepSize ; x++)
      {
        //Motor 1
        digitalWrite(stepPin_M1, HIGH);
        digitalWrite(stepPin_M2, HIGH);
        digitalWrite(stepPin_M3, HIGH);
        delayMicroseconds(pitchVals[pitchVal[i]]);
        
        digitalWrite(stepPin_M1, HIGH);
        digitalWrite(stepPin_M2, LOW);
        digitalWrite(stepPin_M3, LOW);
        delayMicroseconds(pitchVals[pitchVal[i]]);
       }
    }
    pedalIndex = pedalIndex+1;
    pitchVal[i]=0;
  }
  counter = 0;
}
