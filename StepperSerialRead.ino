#include "pitches.h"
#define MAX_MILLIS_TO_WAIT 1000  //or whatever
#define MAXBYTES 10

// Note period definitions
#define silence 0
#define A1 18181
#define As1 17161
#define B1 16196
#define C2 15288
#define Cs2 14430
#define D2 13620
#define Ds2 12856
#define E2 12134
#define F2 11453
#define Fs2 10810
#define G2 10204
#define Gs2 9631
#define A2 9090
#define As2 8580
#define B2 8099
#define C3 7644
#define Cs3 7215
#define D3 6810
#define Ds3 6428
#define E3 6067
#define F3 5727
#define Fs3 5405
#define G3 5102
#define Gs3 4815
#define A3 4545
#define As3 4290
#define B3 4049
#define C4 3822
#define Cs4 3607
#define D4 3405
#define Ds4 3214
#define E4 3033
#define F4 2863
#define Fs4 2702
#define G4 2551
#define Gs4 2407
#define A4 2272
#define As4 2145
#define B4 2024
#define C5 1911
#define Cs5 1803
#define D5 1702
#define Ds5 1607
#define E5 1516
#define F5 1431
#define Fs5 1351
#define G5 1275
#define Gs5 1203
#define A5 1136
#define As5 1072
#define B5 1012
#define C6 955
#define Cs6 901
#define D6 851
#define Ds6 803
#define E6 758
#define F6 715
#define Fs6 675
#define G6 637
#define Gs6 601
#define A6 568
#define As6 536
#define B6 506

// #define E5 = 10
// Three parts, divided into separate arrays for note periods + note lengths
const int mario1F[] PROGMEM = {
 Fs4, Fs4, silence, Fs4, silence, Fs4, Fs4, silence, G4, silence, G4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, E5, Ds5, D5, B4, G3, C5, silence, F5, silence, F5, F5, silence, G3, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, Gs4, silence, F4, silence, E4, silence, G3, G3, silence, C3, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, E5, Ds5, D5, B4, G3, C5, silence, F5, silence, F5, F5, silence, G3, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, Gs4, silence, F4, silence, E4, silence, G3, G3, silence, C3, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, G4, G3, silence, C3, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Fs4, Fs4, silence, Fs4, silence, Fs4, Fs4, silence, G4, silence, G4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, silence, F5, silence, F5, silence, F5, silence, E5, B3, silence, C5, A4, G3, F4, E4, C4, G3, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, D5, silence, D5, D5, silence, C5, silence, B4, silence, G4, E4, G3, E4, C4, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, silence, F5, silence, F5, silence, F5, silence, E5, B3, silence, C5, A4, G3, F4, E4, C4, G3, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, D5, silence, D5, D5, silence, C5, silence, B4, silence, G4, E4, G3, E4, C4, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, G4, G3, silence, C3, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Fs4, Fs4, silence, Fs4, silence, Fs4, Fs4, silence, G4, silence, G4, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, silence, F5, silence, F5, silence, F5, silence, E5, B3, silence, C5, A4, G3, F4, E4, C4, G3, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, D5, silence, D5, D5, silence, C5, silence, B4, silence, G4, E4, G3, E4, C4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, E5, Ds5, D5, B4, G3, C5, silence, F5, silence, F5, F5, silence, G3, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, Gs4, silence, F4, silence, E4, silence, G3, G3, silence, C3, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, E5, Ds5, D5, B4, G3, C5, silence, F5, silence, F5, F5, silence, G3, silence, C3, silence, E5, Ds5, D5, B4, C4, C5, F3, E4, F4, G4, C4, C4, E4, F4, C3, silence, Gs4, silence, F4, silence, E4, silence, G3, G3, silence, C3, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, G4, G3, silence, C3, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Fs4, Fs4, silence, Fs4, silence, Fs4, Fs4, silence, G4, silence, G4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, E4, silence, C4, silence, G3, silence, C4, silence, D4, silence, Cs4, C4, silence, C4, silence, G4, silence, B4, silence, C5, silence, A4, B4, silence, A4, silence, E4, F4, D4, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, silence, F5, silence, F5, silence, F5, silence, E5, B3, silence, C5, A4, G3, F4, E4, C4, G3, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, D5, silence, D5, D5, silence, C5, silence, B4, silence, G4, E4, G3, E4, C4, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, silence, F5, silence, F5, silence, F5, silence, E5, B3, silence, C5, A4, G3, F4, E4, C4, G3, silence, C5, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, D5, silence, D5, D5, silence, C5, silence, B4, silence, G4, E4, G3, E4, C4, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, G4, G3, silence, C3, silence, G2, silence, Gs4, Gs4, silence, Gs4, silence, Gs4, As4, silence, G4, E4, silence, E4, C4, silence, G2, silence, Fs4, Fs4, silence, Fs4, silence, Fs4, Fs4, silence, G4, silence, G4, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, silence, F5, silence, F5, silence, F5, silence, E5, B3, silence, C5, A4, G3, F4, E4, C4, G3, silence, C5, A4, silence, E4, G3, silence, E4, silence, F4, C5, F3, C5, F4, C4, F3, silence, G4, D5, silence, D5, D5, silence, C5, silence, B4, silence, G4, E4, G3, E4, C4
};
const int mario1T[] PROGMEM = {
 152, 149, 149, 148, 154, 149, 148, 149, 154, 446, 154, 468, 159, 293, 148, 303, 148, 303, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 303, 148, 303, 149, 152, 150, 152, 149, 148, 153, 105, 99, 100, 100, 97, 99, 154, 150, 148, 148, 155, 149, 148, 148, 154, 149, 305, 157, 146, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 147, 148, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 148, 154, 157, 146, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 146, 148, 303, 149, 297, 154, 307, 151, 159, 145, 148, 148, 154, 150, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 147, 148, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 150, 148, 154, 157, 146, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 147, 148, 302, 149, 298, 154, 306, 151, 159, 145, 148, 148, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 150, 148, 149, 154, 149, 149, 148, 155, 149, 148, 148, 154, 297, 148, 304, 148, 149, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 149, 148, 154, 149, 148, 160, 159, 149, 148, 148, 155, 149, 148, 149, 154, 446, 154, 468, 159, 292, 148, 304, 148, 303, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 314, 159, 292, 148, 304, 148, 303, 149, 151, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 314, 159, 149, 148, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 105, 98, 100, 101, 97, 103, 105, 99, 100, 148, 148, 154, 149, 148, 150, 157, 149, 148, 159, 159, 149, 149, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 157, 149, 155, 151, 107, 96, 100, 100, 97, 100, 154, 149, 148, 151, 159, 441, 154, 149, 149, 148, 154, 150, 148, 148, 154, 149, 148, 150, 157, 149, 148, 154, 105, 98, 100, 101, 97, 103, 105, 98, 100, 148, 149, 154, 149, 148, 150, 157, 149, 148, 159, 159, 149, 149, 148, 154, 150, 148, 148, 154, 149, 148, 150, 157, 149, 148, 154, 157, 149, 154, 151, 107, 96, 100, 100, 97, 100, 154, 149, 148, 151, 159, 441, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 149, 148, 154, 149, 148, 149, 154, 149, 149, 148, 154, 149, 148, 148, 154, 298, 148, 304, 148, 148, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 150, 148, 159, 159, 149, 149, 148, 154, 149, 148, 148, 154, 446, 154, 468, 159, 149, 149, 148, 154, 150, 148, 148, 154, 149, 148, 150, 157, 149, 148, 154, 105, 98, 100, 101, 97, 103, 105, 98, 100, 148, 149, 154, 149, 148, 150, 157, 149, 148, 159, 159, 149, 149, 148, 154, 150, 148, 148, 154, 149, 148, 150, 157, 149, 148, 154, 157, 149, 154, 151, 107, 96, 100, 100, 97, 100, 154, 149, 148, 148, 154, 446, 154, 298, 148, 304, 148, 303, 149, 151, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 313, 159, 293, 148, 304, 148, 302, 149, 152, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 305, 157, 146, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 146, 148, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 148, 154, 157, 147, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 146, 148, 303, 149, 297, 154, 307, 151, 159, 144, 148, 149, 154, 149, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 146, 148, 148, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 148, 154, 157, 147, 148, 151, 159, 149, 148, 148, 154, 149, 148, 148, 154, 149, 148, 150, 157, 146, 148, 303, 149, 297, 154, 307, 151, 159, 144, 148, 149, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 148, 148, 154, 149, 148, 149, 154, 149, 149, 148, 154, 149, 148, 148, 154, 298, 148, 304, 148, 148, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 150, 148, 159, 159, 149, 149, 148, 154, 149, 148, 148, 154, 446, 154, 468, 159, 293, 148, 304, 148, 302, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 304, 148, 302, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 149, 149, 148, 154, 150, 148, 148, 154, 149, 148, 150, 157, 149, 148, 154, 105, 98, 100, 101, 97, 103, 105, 98, 100, 148, 149, 154, 149, 148, 150, 157, 149, 148, 159, 159, 149, 149, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 157, 149, 154, 151, 107, 96, 100, 100, 97, 100, 154, 149, 148, 151, 159, 441, 154, 149, 148, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 105, 98, 100, 101, 97, 103, 105, 99, 100, 148, 149, 154, 149, 148, 150, 157, 149, 148, 160, 159, 292, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 157, 149, 155, 151, 107, 96, 100, 100, 97, 100, 154, 149, 148, 151, 159, 441, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 148, 148, 154, 297, 148, 304, 148, 149, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 148, 148, 154, 149, 148, 160, 159, 149, 148, 148, 155, 149, 148, 149, 154, 446, 154, 468, 159, 149, 148, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 105, 99, 100, 101, 97, 103, 105, 98, 100, 148, 149, 154, 149, 148, 150, 157, 149, 148, 160, 159, 149, 148, 148, 154, 149, 148, 149, 154, 149, 148, 150, 157, 149, 148, 154, 157, 149, 155, 151, 107, 96, 100, 100, 97, 100, 154, 149, 148, 148, 154
};

const int mario2F[] PROGMEM = {
  E5, E5, silence, E5, silence, C5, E5, silence, B4, silence, G3, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, G5, silence, G5, G5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, Ds5, silence, D5, silence, C5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, G5, silence, G5, G5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, Ds5, silence, D5, silence, C5, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, C5, C5, silence, C5, silence, C5, D5, E5, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, E5, E5, silence, E5, silence, C5, E5, silence, B4, silence, G3, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, silence, A5, silence, A5, silence, A5, silence, G5, silence, D5, silence, E5, C5, silence, A4, G4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, F5, silence, F5, F5, silence, E5, silence, D5, silence, C5, silence, C3, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, silence, A5, silence, A5, silence, A5, silence, G5, silence, D5, silence, E5, C5, silence, A4, G4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, F5, silence, F5, F5, silence, E5, silence, D5, silence, C5, silence, C3, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, C5, C5, silence, C5, silence, C5, D5, E5, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, E5, E5, silence, E5, silence, C5, E5, silence, B4, silence, G3, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, silence, A5, silence, A5, silence, A5, silence, G5, silence, D5, silence, E5, C5, silence, A4, G4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, F5, silence, F5, F5, silence, E5, silence, D5, silence, C5, silence, C3, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, G5, silence, G5, G5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, Ds5, silence, D5, silence, C5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, G5, silence, G5, G5, silence, G5, Fs5, F5, Ds5, silence, E5, silence, Gs4, A4, C5, silence, A4, C5, D5, silence, Ds5, silence, D5, silence, C5, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, C5, C5, silence, C5, silence, C5, D5, E5, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, E5, E5, silence, E5, silence, C5, E5, silence, B4, silence, G3, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, C5, silence, G4, silence, E4, silence, A4, silence, B4, silence, As4, A4, silence, G4, silence, E5, silence, G5, silence, A5, silence, F5, G5, silence, E5, silence, C5, D5, B4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, silence, A5, silence, A5, silence, A5, silence, G5, silence, D5, silence, E5, C5, silence, A4, G4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, F5, silence, F5, F5, silence, E5, silence, D5, silence, C5, silence, C3, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, silence, A5, silence, A5, silence, A5, silence, G5, silence, D5, silence, E5, C5, silence, A4, G4, silence, E5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, F5, silence, F5, F5, silence, E5, silence, D5, silence, C5, silence, C3, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, C5, C5, silence, C5, silence, C5, D5, E5, silence, C5, C5, silence, C5, silence, C5, D5, silence, E5, C5, silence, A4, G4, silence, E5, E5, silence, E5, silence, C5, E5, silence, B4, silence, G3, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, silence, A5, silence, A5, silence, A5, silence, G5, silence, D5, silence, E5, C5, silence, A4, G4, silence, E5, C5, silence, G4, silence, Gs4, silence, A4, F5, silence, F5, A4, silence, B4, F5, silence, F5, F5, silence, E5, silence, D5, silence, C5, silence, C3
};
const int mario2T[] PROGMEM = {
  152, 149, 149, 148, 154, 149, 148, 149, 154, 446, 154, 468, 159, 293, 148, 303, 148, 303, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 303, 148, 303, 149, 152, 150, 152, 149, 148, 153, 105, 99, 100, 100, 97, 99, 154, 150, 148, 148, 155, 149, 148, 148, 154, 149, 608, 148, 151, 159, 149, 149, 148, 154, 149, 148, 148, 154, 149, 148, 150, 305, 148, 148, 154, 149, 149, 148, 154, 149, 149, 148, 154, 754, 148, 151, 159, 149, 149, 148, 154, 149, 148, 148, 154, 149, 148, 150, 304, 148, 303, 149, 297, 154, 1364, 148, 151, 159, 149, 149, 148, 154, 149, 148, 148, 154, 149, 148, 150, 305, 148, 148, 154, 149, 149, 148, 154, 149, 149, 148, 154, 755, 148, 151, 159, 149, 149, 148, 154, 149, 148, 148, 154, 149, 148, 150, 305, 148, 302, 149, 298, 154, 1060, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 447, 154, 149, 149, 148, 155, 149, 148, 148, 1201, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 149, 148, 154, 457, 159, 149, 148, 148, 155, 149, 148, 149, 154, 446, 154, 468, 159, 292, 148, 304, 148, 303, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 314, 159, 292, 148, 304, 148, 303, 149, 151, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 314, 159, 149, 148, 148, 304, 148, 149, 154, 149, 150, 150, 157, 448, 105, 98, 100, 101, 97, 103, 105, 99, 100, 100, 97, 99, 154, 149, 151, 150, 157, 453, 159, 149, 149, 148, 304, 148, 149, 154, 149, 150, 150, 157, 448, 157, 149, 155, 151, 107, 96, 100, 100, 97, 100, 154, 454, 159, 441, 154, 149, 149, 148, 304, 148, 148, 154, 149, 150, 150, 157, 448, 105, 98, 100, 101, 97, 103, 105, 98, 100, 100, 97, 100, 154, 149, 150, 150, 157, 454, 159, 149, 149, 148, 304, 148, 148, 154, 149, 150, 150, 157, 449, 157, 149, 154, 151, 107, 96, 100, 100, 97, 100, 154, 455, 159, 441, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 149, 148, 154, 446, 154, 149, 149, 148, 154, 149, 148, 148, 1200, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 457, 159, 149, 149, 148, 154, 149, 148, 148, 154, 446, 154, 468, 159, 149, 149, 148, 304, 148, 148, 154, 149, 151, 150, 157, 448, 105, 98, 100, 101, 97, 103, 105, 98, 100, 100, 97, 100, 154, 149, 150, 150, 157, 454, 159, 149, 149, 148, 304, 148, 148, 154, 149, 150, 150, 157, 449, 157, 149, 154, 151, 107, 96, 100, 100, 97, 100, 154, 446, 154, 446, 154, 298, 148, 304, 148, 303, 149, 151, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 313, 159, 293, 148, 304, 148, 302, 149, 152, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 608, 148, 151, 159, 149, 149, 148, 155, 149, 148, 148, 155, 149, 148, 150, 304, 148, 148, 154, 149, 149, 148, 154, 149, 149, 148, 154, 755, 148, 151, 159, 149, 148, 148, 155, 149, 148, 148, 155, 149, 148, 150, 304, 148, 303, 149, 297, 154, 1364, 148, 151, 159, 149, 149, 148, 154, 149, 148, 148, 154, 149, 148, 150, 304, 148, 148, 154, 149, 148, 148, 155, 149, 148, 148, 154, 755, 148, 151, 159, 149, 148, 148, 155, 149, 148, 148, 155, 149, 148, 150, 304, 148, 303, 149, 297, 154, 1061, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 148, 148, 154, 446, 154, 149, 149, 148, 154, 149, 148, 148, 1200, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 457, 159, 149, 149, 148, 154, 149, 148, 148, 154, 446, 154, 468, 159, 293, 148, 304, 148, 302, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 304, 148, 302, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 149, 149, 148, 304, 148, 148, 154, 149, 150, 150, 157, 449, 105, 98, 100, 101, 97, 103, 105, 98, 100, 100, 97, 100, 154, 149, 150, 150, 157, 454, 159, 149, 149, 148, 303, 148, 149, 154, 149, 150, 150, 157, 449, 157, 149, 154, 151, 107, 96, 100, 100, 97, 100, 154, 455, 159, 441, 154, 149, 148, 148, 304, 148, 149, 154, 149, 150, 150, 157, 448, 105, 98, 100, 101, 97, 103, 105, 99, 100, 100, 97, 100, 154, 149, 151, 150, 157, 454, 159, 292, 148, 304, 148, 149, 154, 149, 150, 150, 157, 448, 157, 149, 155, 151, 107, 96, 100, 100, 97, 100, 154, 454, 159, 441, 154, 149, 149, 148, 154, 149, 148, 148, 154, 149, 149, 148, 154, 446, 154, 149, 149, 148, 154, 149, 148, 148, 1201, 154, 149, 148, 148, 155, 149, 148, 149, 154, 149, 148, 148, 154, 457, 159, 149, 148, 148, 155, 149, 148, 149, 154, 446, 154, 468, 159, 149, 148, 148, 304, 148, 149, 154, 149, 150, 150, 157, 448, 105, 99, 100, 101, 97, 103, 105, 98, 100, 100, 97, 100, 154, 149, 150, 150, 157, 454, 159, 149, 148, 148, 304, 148, 149, 154, 149, 150, 150, 157, 448, 157, 149, 155, 151, 107, 96, 100, 100, 97, 100, 154, 446, 154
};

const int mario3F[] PROGMEM = {
  D3, D3, silence, D3, silence, D3, D3, silence, G5, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, G3, silence, C4, silence, F3, silence, E3, silence, C4, silence, C6, silence, C6, C6, silence, G3, silence, C4, silence, F3, silence, Gs3, silence, As3, silence, C4, silence, G3, silence, C4, silence, F3, silence, E3, silence, C4, silence, C6, silence, C6, C6, silence, G3, silence, C4, silence, F3, silence, Gs3, silence, As3, silence, C4, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, Gs2, silence, Ds3, silence, Gs3, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, D3, D3, silence, D3, silence, D3, D3, silence, G5, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, D3, silence, F3, G3, silence, F5, silence, G3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, G3, silence, G3, G3, silence, A3, silence, B3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, D3, silence, F3, G3, silence, F5, silence, G3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, G3, silence, G3, G3, silence, A3, silence, B3, silence, C4, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, Gs2, silence, Ds3, silence, Gs3, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, D3, D3, silence, D3, silence, D3, D3, silence, G5, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, D3, silence, F3, G3, silence, F5, silence, G3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, G3, silence, G3, G3, silence, A3, silence, B3, silence, C4, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, G3, silence, C4, silence, F3, silence, E3, silence, C4, silence, C6, silence, C6, C6, silence, G3, silence, C4, silence, F3, silence, Gs3, silence, As3, silence, C4, silence, G3, silence, C4, silence, F3, silence, E3, silence, C4, silence, C6, silence, C6, C6, silence, G3, silence, C4, silence, F3, silence, Gs3, silence, As3, silence, C4, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, Gs2, silence, Ds3, silence, Gs3, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, D3, D3, silence, D3, silence, D3, D3, silence, G5, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, G3, silence, E3, silence, C3, silence, F3, silence, G3, silence, Fs3, F3, silence, E3, silence, C4, silence, E4, silence, F4, silence, D4, E4, silence, C4, silence, A3, B3, G3, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, D3, silence, F3, G3, silence, F5, silence, G3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, G3, silence, G3, G3, silence, A3, silence, B3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, D3, silence, F3, G3, silence, F5, silence, G3, silence, C4, silence, C3, A4, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, G3, silence, G3, G3, silence, A3, silence, B3, silence, C4, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, Gs2, silence, Ds3, silence, Gs3, silence, Gs2, silence, Ds3, silence, Gs3, silence, G3, silence, C3, silence, D3, D3, silence, D3, silence, D3, D3, silence, G5, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, D3, silence, F3, G3, silence, F5, silence, G3, silence, C4, silence, C3, silence, Fs3, silence, C4, silence, F3, silence, C4, silence, G3, silence, G3, G3, silence, A3, silence, B3, silence, C4
};
const int mario3T[] PROGMEM = {
  152, 149, 149, 148, 154, 149, 148, 149, 154, 1068, 159, 293, 148, 303, 148, 303, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 303, 148, 303, 149, 152, 150, 152, 149, 148, 153, 105, 99, 100, 100, 97, 99, 154, 150, 148, 148, 155, 149, 148, 148, 154, 149, 759, 151, 1055, 148, 304, 148, 604, 148, 452, 148, 154, 149, 149, 148, 154, 906, 151, 1054, 148, 304, 148, 456, 148, 303, 149, 297, 154, 1515, 151, 1055, 148, 303, 148, 605, 148, 452, 148, 154, 149, 149, 148, 154, 906, 151, 1055, 148, 304, 148, 456, 148, 302, 149, 298, 154, 1060, 154, 298, 148, 304, 148, 148, 154, 298, 148, 601, 154, 297, 148, 304, 148, 1349, 154, 297, 148, 304, 148, 149, 154, 298, 148, 611, 159, 149, 148, 148, 155, 149, 148, 149, 154, 1068, 159, 292, 148, 304, 148, 303, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 314, 159, 292, 148, 304, 148, 303, 149, 151, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 150, 148, 148, 154, 149, 149, 148, 154, 149, 314, 159, 292, 148, 304, 148, 149, 154, 450, 157, 448, 157, 296, 150, 157, 247, 97, 99, 154, 451, 157, 453, 159, 293, 148, 304, 148, 149, 154, 450, 157, 448, 157, 301, 151, 107, 96, 100, 100, 97, 100, 154, 1054, 154, 298, 148, 304, 148, 148, 154, 451, 157, 448, 157, 296, 150, 157, 246, 97, 100, 154, 450, 157, 454, 159, 293, 148, 304, 148, 148, 154, 450, 157, 449, 157, 300, 151, 107, 96, 100, 100, 97, 100, 154, 1055, 154, 297, 148, 304, 148, 149, 154, 298, 148, 600, 154, 298, 148, 303, 148, 1349, 154, 298, 148, 304, 148, 148, 154, 298, 148, 611, 159, 149, 149, 148, 154, 149, 148, 148, 154, 1068, 159, 293, 148, 304, 148, 148, 154, 451, 157, 448, 157, 296, 150, 157, 246, 97, 100, 154, 450, 157, 454, 159, 293, 148, 304, 148, 148, 154, 450, 157, 449, 157, 300, 151, 107, 96, 100, 100, 97, 100, 154, 1046, 154, 298, 148, 304, 148, 303, 149, 151, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 313, 159, 293, 148, 304, 148, 302, 149, 152, 150, 153, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 154, 149, 149, 148, 154, 149, 760, 151, 1054, 148, 304, 148, 604, 148, 453, 148, 154, 149, 149, 148, 154, 906, 151, 1054, 148, 304, 148, 456, 148, 303, 149, 297, 154, 1516, 151, 1054, 148, 304, 148, 604, 148, 452, 148, 155, 149, 148, 148, 154, 906, 151, 1054, 148, 304, 148, 456, 148, 303, 149, 297, 154, 1061, 154, 297, 148, 304, 148, 149, 154, 297, 148, 601, 154, 298, 148, 303, 148, 1349, 154, 298, 148, 304, 148, 148, 154, 298, 148, 611, 159, 149, 149, 148, 154, 149, 148, 148, 154, 1068, 159, 293, 148, 304, 148, 302, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 304, 148, 302, 149, 152, 150, 152, 149, 148, 154, 105, 98, 100, 100, 97, 100, 154, 149, 148, 148, 155, 149, 148, 148, 154, 149, 313, 159, 293, 148, 304, 148, 148, 154, 450, 157, 449, 157, 296, 150, 157, 246, 97, 100, 154, 450, 157, 454, 159, 293, 148, 303, 148, 149, 154, 450, 157, 449, 157, 300, 151, 107, 96, 100, 100, 97, 100, 154, 1055, 154, 297, 148, 304, 148, 149, 154, 450, 157, 448, 157, 296, 150, 157, 247, 97, 100, 154, 450, 157, 454, 159, 149, 149, 148, 304, 148, 149, 154, 450, 157, 448, 157, 301, 151, 107, 96, 100, 100, 97, 100, 154, 1054, 154, 298, 148, 304, 148, 148, 154, 298, 148, 600, 154, 298, 148, 304, 148, 1349, 154, 297, 148, 304, 148, 149, 154, 297, 148, 612, 159, 149, 148, 148, 155, 149, 148, 149, 154, 1068, 159, 292, 148, 304, 148, 149, 154, 450, 157, 448, 157, 296, 150, 157, 246, 97, 100, 154, 450, 157, 454, 159, 292, 148, 304, 148, 149, 154, 450, 157, 448, 157, 301, 151, 107, 96, 100, 100, 97, 100, 154
};

int arrayGet (int *arr, int index) {
  return pgm_read_word_near(arr + index);
}

// This class maintains the play head state of a single part of the song. The `tick()` method can be called
// as many times per second as possible, given the current time in micros. It is responsible for moving the 
// stepper motor assigned to it ahead at precisely the right time to make it resonate notes
class PlayThread {
  public:
    PlayThread(int pin, int totalSteps, int time[], int freq[]);
    void timeslice(unsigned long now);
  private:
    void tick();
    
    int pin;
    int* time;
    int* freq;
    int totalSteps;
    int currentStep;
    unsigned long nextStepTime;
    unsigned long nextTickTime;
};

PlayThread::PlayThread(int _pin, int _totalSteps, int _time[], int _freq[]) {
  pin = _pin;
  time = _time;
  freq = _freq;
  totalSteps = _totalSteps;
  currentStep = -1;
  nextStepTime = 0;
  nextTickTime = 0;
}

void PlayThread::timeslice(unsigned long now) {
  if (now > nextStepTime) {
    // Time for next action
    currentStep += 1;
    if (currentStep > totalSteps) { 
      currentStep = 0;
    }
    
    unsigned long timeStep = ((unsigned long)pgm_read_word_near(time + currentStep))*1000L;
    int freqStep = pgm_read_word_near(freq + currentStep);
    
    nextStepTime = now + timeStep;
    if (freqStep == 0) {
      nextTickTime = nextStepTime;
    } else {
      tick();
      nextTickTime = micros() + (unsigned long)freqStep;
    }
    // Serial.println("---");
    // Serial.println(currentStep);
  } else if (now > nextTickTime) {
    // Continue existing action, time to step motor
    int freqStep = pgm_read_word_near(freq + currentStep);
    tick();
    nextTickTime = micros() + (unsigned long)freqStep;
  }
}

void PlayThread::tick() {
  digitalWrite(pin, HIGH);
  delayMicroseconds(5);
  digitalWrite(pin, LOW);
}

unsigned long starttime;

// Define pin connections & motor's steps per revolution
const int dirPin_M1 = 5;
const int stepPin_M1 = 2;

const int dirPin_M2 = 6;
const int stepPin_M2 = 3;

const int dirPin_M3 = 7;
const int stepPin_M3 = 4;

const int stepsPerRevolution = 200;
int pitchVal[5];
int i = 0;
byte bytes[5];
String ReceivedString[5];
char midMarker = 'n';
char endMarker = 'A';
String input;

char receivedNote[MAXBYTES];
bool newData = false;
int iCount = 0;
int SuperMario =0;

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

// Three concurrent parts
PlayThread thread1(stepPin_M1, 1128, mario1T, mario1F);
PlayThread thread2(stepPin_M2, 986, mario2T, mario2F);
PlayThread thread3(stepPin_M3, 754, mario3T, mario3F);

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
}
void loop()
{
  // Set motor direction clockwise
  digitalWrite(dirPin_M1, HIGH);
  digitalWrite(dirPin_M2, HIGH);
  digitalWrite(dirPin_M3, HIGH);
  if(SuperMario==0)
  {
    starttime = millis();
  
    //recWithEndMark();
    while(Serial.available()!=0 && ((millis() - starttime) < MAX_MILLIS_TO_WAIT) )
    {
       String last_state = input;
       //Serial.println("serial available");
      input = Serial.readStringUntil('A');
      Serial.print("Data Received: ");
      Serial.print(input);
      Serial.println("");
  
  
      //Serial.print("Pitch Value: ");
      //Serial.print("[");
      for(int i= 0; i<5; i++)
      {
        if(last_state == input)
        {
          pitchVal[i]=0;
        }
        else
        {
          pitchVal[i] = getValue(input,'n',i).toInt();
        }
        //Serial.print(pitchVal[i]);
        //Serial.print(",");
      }
      //Serial.print("]");
      //Serial.println("");
    }
    /*
    while ( (Serial.available()<5) && ((millis() - starttime) < MAX_MILLIS_TO_WAIT) )
    {      
          // hang in this loop until we either get 9 bytes of data or 1 second
          // has gone by
    }
    if(Serial.available() < 5)
    {
                // the data didn't come in - handle that problem here
          Serial.println("ERROR - Didn't get 5 bytes of data!");
    }
    else
    {
          Serial.print("[");
          for(int n=0; n<5; n++)
          {
            pitchVal[n] = Serial.read(); // Then: Get them. 
            Serial.print(pitchVal[n]);
            Serial.print(",");
          }
          Serial.print("]");
          Serial.println("");
    }*/
    
    for(int i=0; i<5; i++)
    {
      if(pitchVal[i]<72 || pitchVal[i]>100)
      {
        pitchVal[i]=0;
      }
      // Spin motor slowly
      else if(pitchVal[i]>71 && pitchVal[i]<101)
      {
        if(i==5)
        {
          break;
        }
        
        if(i==1)
        {
          for(int x = 0; x < 4*stepsPerRevolution/(127-int(pitchVal[i])); x++)
          {
            //Motor 1
            digitalWrite(stepPin_M1, HIGH);
            delayMicroseconds(pitchVals[pitchVal[i]]);
            digitalWrite(stepPin_M1, LOW);
            delayMicroseconds(pitchVals[pitchVal[i]]);
          }
        }
        if(i==2)
        {
          for(int x = 0; x < 4*stepsPerRevolution/(127-int(pitchVal[i])); x++)
          {
            //Motor 2
            digitalWrite(stepPin_M2, HIGH);
            delayMicroseconds(pitchVals[pitchVal[i]]);
            digitalWrite(stepPin_M2, LOW);
            delayMicroseconds(pitchVals[pitchVal[i]]);
          }
        }
        if(i==3)
        {
          for(int x = 0; x < 4*stepsPerRevolution/(127-int(pitchVal[i])); x++)
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
        }
      }
      delay(10);
      pitchVal[i]=0;
    }
    newData =false;
  }
  else if(SuperMario==1)
  {
    //xSerial.println("Player Piano Song 1 Chosen");
    // The faster we do this the better the notes sound, precision is key
    thread1.timeslice(micros());
    thread2.timeslice(micros());
    thread3.timeslice(micros());
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
