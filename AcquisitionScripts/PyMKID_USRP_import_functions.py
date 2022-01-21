from __future__ import division
## Import the relevant modules
import sys

## Try to read in the USRP modules
## Exit out if you can't after adjusting path
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../DeviceControl/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")
        exit()

def noise_run(rate,freq,front_end,tones,lapse,decimation,tx_gain,rx_gain,vna,mode,pf,trigger,amplitudes,delay):

    if trigger is not None:
        try:
            trigger = eval('u.'+trigger+'()')
        except SyntaxError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""
        except AttributeError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""

    noise_filename = u.Get_noise(tones, measure_t = lapse, rate = rate, decimation = decimation, amplitudes = amplitudes,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = delay*1e-9,
                              pf_average = pf, tx_gain = tx_gain, rx_gain = rx_gain, mode = mode, trigger = trigger)
    if vna is not None:
        u.copy_resonator_group(vna, noise_filename)

    return noise_filename

def vna_run(tx_gain,rx_gain,iter,rate,freq,front_end,f0,f1,lapse,points,ntones,delay_duration,delay_over='null',output_filename=None):

    if delay_over != 'null':
        delay = delay_over
        u.set_line_delay(rate, delay_over*1e9)
    else:
        try:
            if u.LINE_DELAY[str(int(rate/1e6))]:
                delay = u.LINE_DELAY[str(int(rate/1e6))]*1e-9
        except KeyError:
            print("Cannot find line delay. Measuring line delay before VNA:")
            delay_filename=None
            if(output_filename != None):
                delay_filename = output_filename + '_delay'
            filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=tx_gain, rx_gain=rx_gain, output_filename=delay_filename, compensate = True, duration = delay_duration)

            print("Analyziung line delay file...")
            delay = u.analyze_line_delay(filename,False)
            print("Done.")

            print("Writing line delay to file...")
            u.write_delay_to_file(filename, delay)
            print("Done.")

            print("Loading line delay from file...")
            u.load_delay_from_file(filename)
            print("Done.")

    if ntones ==1:
        ntones = None

    print("Starting single VNA run...")
    vna_filename = u.Single_VNA(start_f = f0, last_f = f1, measure_t = lapse, n_points = points, tx_gain = tx_gain, rx_gain= rx_gain, Rate=rate, decimation=True, RF=freq, Front_end=front_end,
               Device=None, output_filename=output_filename, Multitone_compensation=ntones, Iterations=iter, verbose=False)
    print("Done.")

    return vna_filename,delay
