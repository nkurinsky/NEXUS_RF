#include <Wire.h>
#include "si5351.h"

// Pin IDs
const byte flipFlopInPin = 2;  // "TRIG OUT" Pin
const byte trigInPin     = 3;  // "TRIG IN" Pin
const byte ledPin        = 13; // Control the LED state

// TPL0401B-10DCKR parameters
#define pot_address 0x2F
int r_int = 127;

// Si5351A-B-GT parameters
Si5351 si5351;
unsigned long long MHz = 100000000ULL;
unsigned long long freq = 0.05*MHz; // default = 0.05 MHz, 1 us p.w.
// In unit of 0.01 Hz, 1 MHz means 500 ns pulse width
// PW[us] = 0.5 / freq[MHz] due to 50% duty cycle of Si5351 chip

// Communication variables
String a;
double f_temp, r_temp;

// Control and timing parameters
double burst_rate  = 250.0;  // Default to 250 Hz

int delay_ms = (int) 1000 / burst_rate;
int delay_on_us = 10;

// Enable laser
bool enable = false;
bool user_allow = false;

// Define a counter for the number of pulses 
// sent in a singe 1PPS cycle
int counter = 0;

void initialize_si5351(unsigned long long new_freq)
{
  si5351.reset();
  si5351.output_enable(SI5351_CLK0, 0);
  si5351.output_enable(SI5351_CLK1, 0);
  si5351.set_freq(new_freq, SI5351_CLK0);
  si5351.set_freq(new_freq, SI5351_CLK1);
  si5351.set_phase(SI5351_CLK0, 0);
  si5351.set_phase(SI5351_CLK1, 0);
  si5351.output_enable(SI5351_CLK0, 1);
  si5351.output_enable(SI5351_CLK1, 1);
  si5351.pll_reset(SI5351_PLLA);
}

void update_pot(int new_r)
{
  Wire.beginTransmission(pot_address);
  Wire.write(new_r);
  Wire.endTransmission();
}

void enable_laser()
{
  // Called by interrupt of rising edge on TRIG_IN
  enable = true;
}

void disable_laser()
{
  // Called by fire_laser when a full second of pulses has been sent
  enable  = false;
  counter = 0;
}

void setup()
{
  // Define our pins
  pinMode(flipFlopInPin, OUTPUT); // TRIG_OUT pin goes to flip-flop input
  pinMode(trigInPin, INPUT);      // TRIG_IN pin (don't need a pull-up)
  pinMode(ledPin, OUTPUT);        // LED pin

  // Watch the TRIG_IN pin for a rising edge
  // which enables the laser output in the loop
  attachInterrupt(digitalPinToInterrupt(trigInPin), enable_laser, RISING);

  // Connect to the clock
  Serial.begin(115200);
  bool i2c_found = si5351.init(SI5351_CRYSTAL_LOAD_8PF, 0, 0);

  // Initialize the clock to default frequency
  initialize_si5351(freq);
  si5351.update_status();

  // Update the potentiometer to default resistance
  update_pot(r_int);
}

void read_serial()
{
  // Get the buffer contents
  a = Serial.readStringUntil('\n');
  int alen = a.length();

  // Reply to queries
  if (a.substring(alen-2,alen-1) == "?")
  {
    // Standard IDENTITY query
    if (a == "*IDN?") Serial.println("Laser pulser");

    // Get set pulse width
    if (a.substring(0,1) == "F") 
    {
      Serial.print(f_temp, 2);
      Serial.println(" MHz");
    }

    // Get set burst rate
    if (a.substring(0,1) == "B") 
    {
      Serial.print(burst_rate, 2);
      Serial.println(" Hz");
    }

    // Get set resistance
    if (a.substring(0,1) == "B") 
    {
      Serial.println(r_temp, 0);
    }
    
  }

  // LED control commands
  else if (a == "*LED1")
  {
    digitalWrite(ledPin, HIGH);
    Serial.println("LED:1");
  }
  else if (a == "*LED0")
  {
    digitalWrite(ledPin, LOW);
    Serial.println("LED:0");
  }

  // Laser user enable switch
  else if (a == "ON")
  { 
    Serial.println("Enabling laser output");
    user_allow = true;
  }
  else if (a == "OFF")
  {
    Serial.println("Disabling laser output");
    user_allow = false;
  }

  // Detect the command by the first charactor of the string
  // Choices are:
  // FXX.XX\n -- XX.XX from 00.10 to 99.99, in MHz.
  //             This sets the pulse frequency.
  //             1.00 MHz means 500 ns pulse width.
  //             Default is 0.05 MHz.
  // BXX.XX\n -- XX.XX from 00.01 to 999.9, in Hz.
  //             This sets the burst frequency. 1 Hz meaning one pulse per second.
  //             Default is 10.00 Hz.
  // RXXX\n   -- XXX from 000 to 127.
  //             This sets the resistance thus the laser power.
  //             Default is 127.
  else
  {
    // Set the frequency of the clock
    // this sets the pulse width given a 50% duty cycle
    if (a.substring(0, 1) == "F")
    {
      Serial.println("Setting pulse frequency (width) to: "+a.substring(1, 6)+" MHz");
      f_temp = a.substring(1, 6).toDouble();
      unsigned long long f = MHz * f_temp;
      initialize_si5351(f);
    }

    // Set the rate to toggle the flip-flop
    // This is the rate at which pulses are delivered to the laser
    else if (a.substring(0, 1) == "B")
    {
      Serial.println("Setting burst rate to: "+a.substring(1, 6)+" Hz");
      burst_rate = a.substring(1, 6).toDouble();
      if (burst_rate < 5.0)
      {
        Serial.println("Burst rate too low, setting to 5 Hz");
        burst_rate = 5.0;
      }
      delay_ms = (int) 1000 / burst_rate;
    }

    // Update the potentiometer's resistance
    // to change the power delivered to the laser
    else if (a.substring(0, 1) == "R")
    {
      Serial.println("Setting R to: "+a.substring(1, 4));
      r_temp = a.substring(1, 4).toInt();
      update_pot(r_temp);
    }
  }
}

void fire_laser()
{
  // Set the trigger out pin HIGH
  // This is also the data input 
  // for the flip-flop
  if (user_allow)
  {
    digitalWrite(flipFlopInPin, HIGH);
  }

  // Increment the pulse counter
  counter++;

  // Wait a fixed amount before turning 
  // the flip flop data in pin LOW
  delayMicroseconds(delay_on_us);
  digitalWrite(flipFlopInPin, LOW);

  // Check if we've sent a full seconds worth of pulses
  // If so, turn off the laser and wait for another rising edge on TRIG_IN
  // Serial.println(String(counter));
  if (counter == (int)(burst_rate-5)) disable_laser();
}

void loop()
{
  // Read any messages from the PC
  if (Serial.available() > 0) read_serial();

  // If the laser has been turned on
  if (enable)
  {
    // Send a pulse through the flip-flop
    // then set the trigger out line low again
    fire_laser();
    
    // Now wait the burst frequency in ms
    // before executing the loop again
    delay(delay_ms);
  }

  // If the laser is off just hang out
  else delay((int)delay_ms/10);
}
