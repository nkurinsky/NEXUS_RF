#include "si5351.h"
#include "Wire.h"

#define pot_address 0x2F
Si5351 si5351;

String a;

int onoff = 0; // Default to off.
double freq_in = 1.0;
double burst_rate = 10.0;
int r_int = 127;
int delay_ms = 100;
int delay_on_us = 10;
unsigned long long freq = 200000000ULL; // In unit of 0.01 Hz, 1 MHz means 1 us on time

void setup()
{
  pinMode(2, OUTPUT);

  bool i2c_found;
  Serial.begin(115200);
  i2c_found = si5351.init(SI5351_CRYSTAL_LOAD_8PF, 0, 0);
  freq = 200000000ULL; // In unit of 0.01 Hz, 1 MHz means 1 us on time
  si5351.reset();
  si5351.output_enable(SI5351_CLK0, 1);
  si5351.output_enable(SI5351_CLK1, 1);
  si5351.set_freq(freq, SI5351_CLK0);
  si5351.set_freq(freq, SI5351_CLK1);
  si5351.set_phase(SI5351_CLK0, 0);
  si5351.set_phase(SI5351_CLK1, 0);
  si5351.pll_reset(SI5351_PLLA);

  si5351.update_status();
  Wire.beginTransmission(pot_address);
  Wire.write(r_int);
  Wire.endTransmission();
}

void loop()
{
  if (Serial.available() > 0) {
    a = Serial.readStringUntil('\n'); //-- readString is "read till timeout".
    //Changing to readStringUntil('\n'). All commands sent needs to end with \n.
    if (a == "*IDN?")
    {
      Serial.println("Laser pulser");
    }
    else if (a == "*LED1")
    {
      digitalWrite(13, HIGH);
      Serial.println("LED1");
    }
    else if (a == "*LED0")
    {
      digitalWrite(13, LOW);
      Serial.println("LED0");
    }
    else
    {
      // Detect the command by the first charactor of the string
      // Choices are:
      // FXX.XX\n -- XX.XX from 00.10 to 99.99, in MHz.
      //             This sets the pulse frequency.
      //             1.00 MHz means 500 ns on time.
      //             Default is 1.00 MHz.
      // BXX.XX\n -- XX.XX from 00.01 to 999.9, in Hz.
      //             This sets the burst frequency. 1 Hz meaning one pulse per second.
      //             Default is 10.00 Hz.
      // RXXX\n   -- XXX from 000 to 127.
      //             This sets the resistance thus the laser power.
      //             Default is 127.
      // ON\n     -- This starts the laser.
      // OFF\n    -- This stops the laser.
      //            Default is Laser off.

      if (a == "ON")
      {
        onoff = 1;
        Serial.println("Turning laser on");
      }
      else if (a == "OFF")
      {
        onoff = 0;
        Serial.println("Turning laser off");
      }
      else if (a.substring(0, 1) == "F")
      {
        freq_in = a.substring(1, 6).toDouble();
        Serial.println("Setting pulse freq (width) to: "+a.substring(1, 6)+" MHz");

        freq = (unsigned long long)200000000ULL * freq_in;
        // In unit of 0.01 Hz, 1 MHz means actually 1 us on time
        // So 200000000ULL is 500 ns on time
        si5351.reset();
        si5351.output_enable(SI5351_CLK0, 0);
        si5351.output_enable(SI5351_CLK1, 0);
        si5351.set_freq(freq, SI5351_CLK0);
        si5351.set_freq(freq, SI5351_CLK1);
        si5351.set_phase(SI5351_CLK0, 0);
        si5351.set_phase(SI5351_CLK1, 0);
        si5351.output_enable(SI5351_CLK0, 1);
        si5351.output_enable(SI5351_CLK1, 1);
        si5351.pll_reset(SI5351_PLLA);
      }
      else if (a.substring(0, 1) == "B")
      {
        Serial.println("Setting burst rate to: "+a.substring(1, 6)+" Hz");
        burst_rate = a.substring(1, 6).toDouble();
        delay_ms = (int) 1000 / burst_rate;
      }
      else if (a.substring(0, 1) == "R")
      {
        Serial.println("Setting R factor to: "+a.substring(1, 4));
        r_int = a.substring(1, 4).toInt();
        Wire.beginTransmission(pot_address);
        Wire.write(r_int);
        Wire.endTransmission();
      }
    }

  }
  if (onoff)
  {
    digitalWrite(2, HIGH);
    delayMicroseconds(delay_on_us);
    digitalWrite(2, LOW);
    delay(delay_ms);
  }
  else
  {
    delay(500);
  }


}
